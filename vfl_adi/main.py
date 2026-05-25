from __future__ import annotations

import argparse
import json

import torch

from . import attack as A
from . import datasets as D
from . import train as T

TRAIN_DEFAULTS = {
    "credit":  dict(epochs=30,  lr=1e-3, batch=128),
    "vehicle": dict(epochs=150, lr=1e-3, batch=64),
    "mnist":   dict(epochs=15,  lr=1e-3, batch=128),
    "nuswide": dict(epochs=30,  lr=1e-3, batch=128),
    "cifar10": dict(epochs=30,  lr=1e-3, batch=128),
}
ATTACK_DEFAULTS = {
    "credit":  dict(iters=20, lr=0.2,  subsample=20, n_samples=500, strength=1.0),
    "vehicle": dict(iters=30, lr=0.1,  subsample=20, n_samples=100, strength=3.0),
    "mnist":   dict(iters=40, lr=0.1,  subsample=40, n_samples=150, strength=2.0),
    "nuswide": dict(iters=40, lr=0.05, subsample=40, n_samples=200, strength=3.0),
    "cifar10": dict(iters=30, lr=0.1,  subsample=20, n_samples=200, strength=3.0),
}


def _device(arg):
    return arg if (arg != "cuda" or torch.cuda.is_available()) else "cpu"


def cmd_prepare(args):
    D.PREPARE[args.dataset]()


def cmd_train(args):
    d = dict(TRAIN_DEFAULTS[args.dataset])
    if args.epochs:
        d["epochs"] = args.epochs
    T.train(args.dataset, device=_device(args.device), **d)


def cmd_baseline(args):
    model = T.load_trained(args.dataset, device=_device(args.device))
    res = A.baseline_dominating_rate(model, args.dataset, _device(args.device), n=args.n)
    print(f"[baseline] {args.dataset}: {json.dumps(res)}")
    return res


def cmd_attack(args):
    dev = _device(args.device)
    model = T.load_trained(args.dataset, device=dev)
    d = dict(ATTACK_DEFAULTS[args.dataset])
    if args.n_samples:
        d["n_samples"] = args.n_samples
    if args.strength:
        d["strength"] = args.strength
    res = A.evaluate_attack(model, args.dataset, dev, mode=args.mode, box=args.box,
                            thresholds=(args.threshold,), **d)
    print(f"[attack] {args.dataset}: {json.dumps(res)}")
    return res


def cmd_all(args):
    dev = _device(args.device)
    print(f"==== {args.dataset} ====")
    D.PREPARE[args.dataset]()
    cmd_train(args)
    model = T.load_trained(args.dataset, device=dev)
    base = A.baseline_dominating_rate(model, args.dataset, dev, n=min(500, args.n))
    print(f"[baseline/Table III] {args.dataset}: {json.dumps(base)}")
    summary = {"dataset": args.dataset, "baseline": base, "attacks": {}}
    d = dict(ATTACK_DEFAULTS[args.dataset])
    if args.n_samples:
        d["n_samples"] = args.n_samples
    if args.strength:
        d["strength"] = args.strength
    for box in ("white", "black"):
        for mode in ("random", "bounded"):
            res = A.evaluate_attack(model, args.dataset, dev, mode=mode, box=box,
                                    thresholds=(95, 99), verbose=False, **d)
            summary["attacks"][f"{box}_{mode}"] = res
            print(f"  {box:5s} {mode:8s}: success@95={res['success'][95]:.1f}%  "
                  f"success@99={res['success'][99]:.1f}%  adp={res['adp']:.1f}%")
    print(f"[summary] {json.dumps(summary)}")
    return summary


def main():
    p = argparse.ArgumentParser(prog="vfl_adi")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("prepare", cmd_prepare), ("train", cmd_train),
                     ("baseline", cmd_baseline), ("attack", cmd_attack),
                     ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.add_argument("--dataset", required=True, choices=list(D.SPECS))
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--epochs", type=int, default=0)
        sp.add_argument("--n", type=int, default=1000, help="baseline #A inputs")
        sp.add_argument("--n_samples", type=int, default=0, help="attack #A inputs")
        sp.add_argument("--mode", default="random", choices=["random", "bounded"])
        sp.add_argument("--box", default="white", choices=["white", "black"],
                        help="white = autograd gradients; black = query-based FDM/SPSA")
        sp.add_argument("--threshold", type=int, default=95, choices=[95, 99])
        sp.add_argument("--strength", type=float, default=0.0,
                        help="bounded-mutation strength (0 = dataset default)")
        sp.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
