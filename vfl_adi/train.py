from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import datasets as D
from .models import HeteroLR, SplitNNCifar, SplitNNMnist, SplitNNNusWide

CKPT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")


def build_model(name: str) -> nn.Module:
    spec = D.SPECS[name]
    da, db = D.dims(name)
    if name == "credit":
        return HeteroLR(da, db, num_logits=1)
    if name == "vehicle":
        return HeteroLR(da, db, num_logits=spec.num_classes)
    if name == "mnist":
        return SplitNNMnist(num_classes=spec.num_classes)
    if name == "nuswide":
        return SplitNNNusWide(da, db, num_classes=spec.num_classes)
    if name == "cifar10":
        return SplitNNCifar(da, db, num_classes=spec.num_classes)
    raise ValueError(name)


def ckpt_path(name: str) -> str:
    os.makedirs(CKPT_ROOT, exist_ok=True)
    return os.path.join(CKPT_ROOT, f"{name}.pth")


@torch.no_grad()
def evaluate(model, name, spec, device, split="test"):
    model.eval()
    xa, xb, y = (t.to(device) for t in D.load(name, split))
    correct, total = 0, 0
    scores = []
    for i in range(0, len(y), 1024):
        out = model(xa[i:i + 1024], xb[i:i + 1024])
        if spec.task == "binary":
            p = torch.sigmoid(out.squeeze(1))
            scores.append(p)
            correct += ((p > 0.5).long() == y[i:i + 1024]).sum().item()
        else:
            correct += (out.argmax(1) == y[i:i + 1024]).sum().item()
        total += len(out)
    acc = 100.0 * correct / total
    if spec.task == "binary":
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y.cpu().numpy(), torch.cat(scores).cpu().numpy())
            return acc, auc
        except Exception:
            return acc, float("nan")
    return acc, None


def train(name, epochs=50, lr=1e-3, batch=128, device="cuda", weight_decay=1e-4):
    device = device if torch.cuda.is_available() else "cpu"
    spec = D.SPECS[name]
    model = build_model(name).to(device)

    xa, xb, y = D.load(name, "train")
    loader = DataLoader(TensorDataset(xa, xb, y), batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    print(f"[train] {name}: {len(y)} samples, {epochs} epochs on {device}")
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xa_b, xb_b, y_b in loader:
            xa_b, xb_b, y_b = xa_b.to(device), xb_b.to(device), y_b.to(device)
            opt.zero_grad()
            out = model(xa_b, xb_b)
            loss = bce(out.squeeze(1), y_b.float()) if spec.task == "binary" else ce(out, y_b)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y_b)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            acc, extra = evaluate(model, name, spec, device)
            msg = f"  epoch {ep:3d}  loss {tot / len(y):.4f}  test acc {acc:.2f}%"
            if extra is not None:
                msg += f"  auc {extra:.4f}"
            print(msg)

    path = ckpt_path(name)
    torch.save(model.state_dict(), path)
    print(f"[train] saved -> {path}")
    return path


def load_trained(name, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    model = build_model(name).to(device)
    model.load_state_dict(torch.load(ckpt_path(name), map_location=device, weights_only=True))
    return model.eval()
