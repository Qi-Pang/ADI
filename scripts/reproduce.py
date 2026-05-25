#!/usr/bin/env python
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vfl_adi import attack as A
from vfl_adi import datasets as D
from vfl_adi import train as T
from vfl_adi.main import ATTACK_DEFAULTS, TRAIN_DEFAULTS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run(dataset, device, do_train, n_attack, boxes=("white", "black")):
    dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"\n===== {dataset} =====", flush=True)
    D.PREPARE[dataset]()
    if do_train:
        T.train(dataset, device=dev, **dict(TRAIN_DEFAULTS[dataset]))
    model = T.load_trained(dataset, device=dev)

    base = A.baseline_dominating_rate(model, dataset, dev, n=500)
    print(f"  baseline: {json.dumps(base)}", flush=True)

    ad = dict(ATTACK_DEFAULTS[dataset])
    if n_attack:
        ad["n_samples"] = n_attack
    attacks = {"white": {}, "black": {}}
    for box in boxes:
        for mode in ("random", "bounded"):
            res = A.evaluate_attack(model, dataset, dev, mode=mode, box=box,
                                    thresholds=(95, 99), verbose=False, **ad)
            attacks[box][mode] = res
            print(f"  {box:5s} {mode:8s}: success@95={res['success'][95]:.1f}%  "
                  f"success@99={res['success'][99]:.1f}%  adp={res['adp']:.1f}%", flush=True)
    return {"dataset": dataset, "baseline": base, "attacks": attacks}


def _box_table(results, box):
    title = "Whitebox" if box == "white" else "Blackbox"
    lines = [f"### {title}", "",
             "| Dataset | Base ADP | Random@95 | Random@99 | Bounded@95 | Bounded@99 |",
             "|---|---|---|---|---|---|"]
    for r in results:
        a = r["attacks"][box]
        if not a:
            continue
        def cell(mode, thr):
            return f"{a[mode]['success'][thr]:.1f}%"
        lines.append(f"| {r['dataset']} | {r['baseline']['adp']:.1f}% | "
                     f"{cell('random', 95)} | {cell('random', 99)} | "
                     f"{cell('bounded', 95)} | {cell('bounded', 99)} |")
    return lines + [""]


def write_table(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = ["# ADI results", "",
             "`success` = % of A test inputs turned into ADIs "
             "(dominated proportion r >= threshold); ADP = mean r.", ""]
    for box in ("white", "black"):
        if any(r["attacks"].get(box) for r in results):
            lines += _box_table(results, box)
    md = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + md)
    print(f"[written] {os.path.join(RESULTS_DIR, 'summary.md')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["credit", "vehicle", "mnist", "cifar10"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--n-attack", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    results = [run(d, args.device, not args.no_train, args.n_attack) for d in args.datasets]
    write_table(results)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
