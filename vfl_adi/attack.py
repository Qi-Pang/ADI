from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import datasets as D


def predict(model, x_a, x_b, spec):
    out = model(x_a, x_b)
    if spec.task == "binary":
        return (torch.sigmoid(out.squeeze(1)) > 0.5).long()
    return out.argmax(1)


@torch.no_grad()
def dominated_proportion(model, x_a, xb_pool, spec, target, batch=1024):
    model.eval()
    dom = tot = 0
    for i in range(0, len(xb_pool), batch):
        xb = xb_pool[i:i + batch]
        xa = x_a.expand(len(xb), *x_a.shape[1:])
        pred = predict(model, xa, xb, spec)
        dom += (pred == target).sum().item()
        tot += len(xb)
    return 100.0 * dom / tot


def baseline_dominating_rate(model, name, device, n=1000, thresholds=(95, 99)):
    spec = D.SPECS[name]
    xa_te, _, y_te = (t.to(device) for t in D.load(name, "test"))
    _, xb_te, _ = (t.to(device) for t in D.load(name, "test"))
    pool = xb_te
    n = min(n, len(xa_te))
    rs = []
    for i in range(n):
        r = dominated_proportion(model, xa_te[i:i + 1], pool, spec, int(y_te[i]))
        rs.append(r)
    rs = np.array(rs)
    out = {f"rate_{t}": float((rs >= t).mean() * 100) for t in thresholds}
    out["adp"] = float(rs.mean())
    out["n"] = n
    return out


def _task_loss(out, target, spec):
    if spec.task == "binary":
        return nn.functional.binary_cross_entropy_with_logits(
            out.squeeze(1), target.float().view(-1))
    return nn.functional.cross_entropy(out, target.view(-1))


def _saliency_scalar(out, spec):
    if spec.task == "binary":
        return torch.sigmoid(out).sum()
    return out.var()


def synthesize_adi(model, x_a0, target, S_b, spec, *, mode="random",
                   iters=30, lr=0.1, w_loss=0.999, w_sal=0.001,
                   lo=None, hi=None, eps=None, momentum=0.0):
    target = torch.as_tensor(target, device=x_a0.device).long()
    x_a = x_a0.clone().detach()
    orig = x_a0.clone().detach()
    v = torch.zeros_like(x_a)
    for k in range(len(S_b)):
        x_b = S_b[k:k + 1].clone().detach().requires_grad_(True)
        for _ in range(iters):
            x_a = x_a.detach().requires_grad_(True)
            if x_b.grad is not None:
                x_b.grad = None
            out = model(x_a, x_b)
            loss1 = _task_loss(out, target, spec)
            g1 = torch.autograd.grad(loss1, x_a, retain_graph=True)[0]
            gvar = torch.autograd.grad(_saliency_scalar(out, spec), x_b,
                                       create_graph=True)[0]
            loss2 = gvar.abs().mean()
            g2 = torch.autograd.grad(loss2, x_a, allow_unused=True)[0]
            g = w_loss * g1 if g2 is None else w_loss * g1 + w_sal * g2
            v = momentum * v + g
            x_a = (x_a - lr * v).detach()
            x_a = _project(x_a, orig, mode, lo, hi, eps)
    return x_a.detach()


def _project(x_a, orig, mode, lo, hi, eps):
    if mode == "bounded" and eps is not None:
        x_a = orig + (x_a - orig).clamp(min=-eps, max=eps)
    if lo is not None:
        x_a = torch.max(torch.min(x_a, hi), lo)
    return x_a


def _influence(out, spec):
    return torch.sigmoid(out).squeeze(1) if spec.task == "binary" else out.var(dim=1)


@torch.no_grad()
def _objective_batch(model, xa, x_b, target, spec, w_loss, w_sal, h):
    n = len(xa)
    xb = x_b.expand(n, *x_b.shape[1:])
    out = model(xa, xb)
    if spec.task == "binary":
        loss = nn.functional.binary_cross_entropy_with_logits(
            out.squeeze(1), target.float().expand(n), reduction="none")
    else:
        loss = nn.functional.cross_entropy(out, target.expand(n), reduction="none")
    obj = w_loss * loss
    if w_sal > 0:
        u = torch.randn_like(xb)
        u = u / (u.flatten(1).norm(dim=1).view(n, *([1] * (xb.dim() - 1))) + 1e-12) * h
        sal = (_influence(model(xa, xb + u), spec) - _influence(out, spec)).abs()
        obj = obj + w_sal * sal
    return obj


@torch.no_grad()
def synthesize_adi_blackbox(model, x_a0, target, S_b, spec, *, mode="random",
                            iters=30, lr=0.05, w_loss=0.999, w_sal=0.001,
                            lo=None, hi=None, eps=None, q=20, h=0.02, momentum=0.5):
    target = torch.as_tensor(target, device=x_a0.device).long()
    x_a = x_a0.clone()
    orig = x_a0.clone()
    feat = x_a.shape[1:]
    v = torch.zeros_like(x_a)
    for k in range(len(S_b)):
        x_b = S_b[k:k + 1]
        for _ in range(iters):
            u = (torch.randint(0, 2, (q, *feat), device=x_a.device,
                               dtype=x_a.dtype) * 2 - 1)
            op = _objective_batch(model, x_a + h * u, x_b, target, spec, w_loss, w_sal, h)
            om = _objective_batch(model, x_a - h * u, x_b, target, spec, w_loss, w_sal, h)
            coef = ((op - om) / (2 * h)).view(q, *([1] * len(feat)))
            g = (coef * u).mean(0, keepdim=True)
            g = g / (g.abs().mean() + 1e-12)
            v = momentum * v + g
            x_a = _project(x_a - lr * v.sign(), orig, mode, lo, hi, eps)
    return x_a


def _bounds(name, spec, device, mode, strength=1.0):
    if spec.image:
        x = D.load(name, "train")[0].to(device)
        lo = torch.zeros(1, *x.shape[1:], device=device)
        hi = torch.ones(1, *x.shape[1:], device=device)
        if mode == "bounded":
            eps = (strength * 3.0 * x.std(0, keepdim=True)).clamp(min=0.3 * strength)
        else:
            eps = None
        return lo, hi, eps
    xa = D.load(name, "train")[0].to(device)
    fmin = xa.min(0, keepdim=True).values
    fmax = xa.max(0, keepdim=True).values
    span = (fmax - fmin).clamp(min=1e-6)
    if mode == "bounded":
        margin = strength * span
        return fmin - margin, fmax + margin, None
    return fmin - 5 * span, fmax + 5 * span, None


def evaluate_attack(model, name, device, *, mode="random", box="white",
                    n_samples=200, subsample=20, iters=30, lr=0.1,
                    thresholds=(95, 99), momentum=0.0, strength=1.0, bb_q=0,
                    verbose=True):
    spec = D.SPECS[name]
    xa_te, xb_te, _ = (t.to(device) for t in D.load(name, "test"))
    _, xb_tr, _ = (t.to(device) for t in D.load(name, "train"))
    pool = xb_te
    S_b = xb_tr[:subsample]
    lo, hi, eps = _bounds(name, spec, device, mode, strength)
    w_loss, w_sal = (0.999, 0.001) if mode == "random" else (0.99, 0.01)
    bb_lr = 0.03 if spec.image else 0.15

    n_samples = min(n_samples, len(xa_te))
    rs = []
    for i in range(n_samples):
        x_a0 = xa_te[i:i + 1]
        target = int(predict(model, x_a0, xb_te[i:i + 1], spec).item())
        if box == "black":
            qd = bb_q or (30 if spec.image else 20)
            adi = synthesize_adi_blackbox(model, x_a0, target, S_b, spec, mode=mode,
                                          iters=iters, lr=bb_lr, w_loss=w_loss,
                                          w_sal=w_sal, lo=lo, hi=hi, eps=eps, q=qd)
        else:
            adi = synthesize_adi(model, x_a0, target, S_b, spec, mode=mode,
                                 iters=iters, lr=lr, w_loss=w_loss, w_sal=w_sal,
                                 lo=lo, hi=hi, eps=eps, momentum=momentum)
        rs.append(dominated_proportion(model, adi, pool, spec, target))
        if verbose and (i + 1) % 25 == 0:
            cur = 100.0 * np.mean(np.array(rs) >= thresholds[0])
            print(f"    [{name}/{box}/{mode}] {i + 1}/{n_samples}  "
                  f"success@{thresholds[0]}={cur:.1f}%  adp={np.mean(rs):.1f}%")
    rs = np.array(rs)
    return {
        "mode": mode, "box": box, "n": n_samples, "adp": float(rs.mean()),
        "success": {int(t): float((rs >= t).mean() * 100) for t in thresholds},
    }
