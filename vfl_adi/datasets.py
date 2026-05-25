from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import requests
import torch

DATA_ROOT = os.environ.get(
    "VFL_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
RAW = os.path.join(DATA_ROOT, "raw")
PROCESSED = os.path.join(DATA_ROOT, "processed")


@dataclass
class DatasetSpec:
    name: str
    task: str
    num_classes: int
    image: bool


SPECS = {
    "credit":  DatasetSpec("credit",  "binary",     1,  False),
    "vehicle": DatasetSpec("vehicle", "multiclass", 4,  False),
    "mnist":   DatasetSpec("mnist",   "multiclass", 10, True),
    "nuswide": DatasetSpec("nuswide", "multiclass", 10, False),
    "cifar10": DatasetSpec("cifar10", "multiclass", 10, False),
}


def _download(url: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    print(f"  downloading {url}")
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    with open(dst, "wb") as f:
        f.write(r.content)
    return dst


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def _save(name: str, **arrays) -> str:
    os.makedirs(PROCESSED, exist_ok=True)
    path = os.path.join(PROCESSED, f"{name}.pt")
    torch.save({k: torch.as_tensor(v) for k, v in arrays.items()}, path)
    return path


def processed_path(name: str) -> str:
    return os.path.join(PROCESSED, f"{name}.pt")


def prepare_credit() -> str:
    out = processed_path("credit")
    if os.path.exists(out):
        return out
    import pandas as pd
    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    zpath = _download(url, os.path.join(RAW, "credit.zip"))
    with zipfile.ZipFile(zpath) as z:
        xls = [n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx"))][0]
        with z.open(xls) as f:
            df = pd.read_excel(io.BytesIO(f.read()), header=1)
    df = df.drop(columns=[c for c in df.columns if str(c).upper() == "ID"])
    y = df.iloc[:, -1].to_numpy().astype(np.int64)
    x = df.iloc[:, :-1].to_numpy().astype(np.float32)
    x = _standardize(x).astype(np.float32)
    xa, xb = x[:, :13], x[:, 13:]

    rng = np.random.RandomState(100)
    perm = rng.permutation(len(y))
    xa, xb, y = xa[perm], xb[perm], y[perm]
    n_test = 2000
    _save("credit",
          xa_train=xa[:-n_test], xb_train=xb[:-n_test], y_train=y[:-n_test],
          xa_test=xa[-n_test:], xb_test=xb[-n_test:], y_test=y[-n_test:])
    print(f"  credit: A{xa.shape[1]} B{xb.shape[1]} train={len(y)-n_test} test={n_test}")
    return out


def prepare_vehicle() -> str:
    out = processed_path("vehicle")
    if os.path.exists(out):
        return out
    url = "https://archive.ics.uci.edu/static/public/149/statlog+vehicle+silhouettes.zip"
    zpath = _download(url, os.path.join(RAW, "vehicle.zip"))
    rows, labels = [], []
    with zipfile.ZipFile(zpath) as z:
        dats = [n for n in z.namelist() if n.endswith(".dat")]
        if not dats:
            inner = [n for n in z.namelist() if n.endswith(".zip")][0]
            with zipfile.ZipFile(io.BytesIO(z.read(inner))) as z2:
                for n in [m for m in z2.namelist() if m.endswith(".dat")]:
                    for line in z2.read(n).decode().splitlines():
                        _parse_vehicle_line(line, rows, labels)
        else:
            for n in dats:
                for line in z.read(n).decode().splitlines():
                    _parse_vehicle_line(line, rows, labels)
    classes = sorted(set(labels))
    cmap = {c: i for i, c in enumerate(classes)}
    x = _standardize(np.array(rows, dtype=np.float32)).astype(np.float32)
    y = np.array([cmap[c] for c in labels], dtype=np.int64)
    xa, xb = x[:, :9], x[:, 9:]

    rng = np.random.RandomState(100)
    perm = rng.permutation(len(y))
    xa, xb, y = xa[perm], xb[perm], y[perm]
    n_test = 100
    _save("vehicle",
          xa_train=xa[:-n_test], xb_train=xb[:-n_test], y_train=y[:-n_test],
          xa_test=xa[-n_test:], xb_test=xb[-n_test:], y_test=y[-n_test:])
    print(f"  vehicle: classes={classes} A9 B9 train={len(y)-n_test} test={n_test}")
    return out


def _parse_vehicle_line(line, rows, labels):
    parts = line.split()
    if len(parts) == 19:
        rows.append([float(v) for v in parts[:18]])
        labels.append(parts[18].lower())


def prepare_mnist() -> str:
    out = processed_path("mnist")
    if os.path.exists(out):
        return out
    from torchvision import datasets as tvd

    def to_arrays(train):
        ds = tvd.MNIST(RAW, train=train, download=True)
        x = ds.data.numpy().astype(np.float32) / 255.0
        y = ds.targets.numpy().astype(np.int64)
        xa = x[:, :, :14][:, None, :, :]
        xb = x[:, :, 14:][:, None, :, :]
        return xa, xb, y

    xa_tr, xb_tr, y_tr = to_arrays(True)
    xa_te, xb_te, y_te = to_arrays(False)
    _save("mnist",
          xa_train=xa_tr, xb_train=xb_tr, y_train=y_tr,
          xa_test=xa_te, xb_test=xb_te, y_test=y_te)
    print(f"  mnist: A{xa_tr.shape[1:]} B{xb_tr.shape[1:]} train={len(y_tr)} test={len(y_te)}")
    return out


def prepare_nuswide(top_k: int = 10) -> str:
    out = processed_path("nuswide")
    if os.path.exists(out):
        return out
    import pandas as pd
    root = os.path.join(RAW, "NUS_WIDE")
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"NUS-WIDE raw data not found at {root}. See README 'NUS-WIDE' for "
            "the download/layout instructions.")

    lbl_dir = os.path.join(root, "Groundtruth", "AllLabels")
    counts = {}
    for fn in os.listdir(lbl_dir):
        lbl = fn[:-4].split("_")[-1]
        counts[lbl] = int(pd.read_csv(os.path.join(lbl_dir, fn), header=None).sum().iloc[0])
    labels = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]]
    print(f"  nuswide top-{top_k} labels: {labels}")

    def load_split(dtype):
        lp = os.path.join(root, "Groundtruth", "TrainTestLabels")
        L = pd.concat([pd.read_csv(os.path.join(lp, f"Labels_{l}_{dtype}.txt"),
                                   header=None, names=[l]) for l in labels], axis=1)
        sel = L[L.sum(axis=1) == 1]
        feat_dir = os.path.join(root, "Low_Level_Features")
        fas = []
        for f in sorted(os.listdir(feat_dir)):
            if f.startswith(f"{dtype}_Normalized"):
                d = pd.read_csv(os.path.join(feat_dir, f), header=None, sep=" ").dropna(axis=1)
                fas.append(d)
        Xa = pd.concat(fas, axis=1).loc[sel.index]
        tags = pd.read_csv(os.path.join(root, "NUS_WID_Tags", f"{dtype}_Tags1k.dat"),
                           header=None, sep="\t").dropna(axis=1).loc[sel.index]
        y = np.argmax(sel.to_numpy(), axis=1).astype(np.int64)
        return Xa.to_numpy(np.float32), tags.to_numpy(np.float32), y

    xa_tr, xb_tr, y_tr = load_split("Train")
    xa_te, xb_te, y_te = load_split("Test")
    xa_tr, xa_te = _standardize(xa_tr).astype(np.float32), _standardize(xa_te).astype(np.float32)
    _save("nuswide",
          xa_train=xa_tr, xb_train=xb_tr, y_train=y_tr,
          xa_test=xa_te, xb_test=xb_te, y_test=y_te)
    print(f"  nuswide: A{xa_tr.shape[1]} B{xb_tr.shape[1]} train={len(y_tr)} test={len(y_te)}")
    return out


def prepare_cifar10(half_size: int = 64, batch: int = 256, device: str = "cuda") -> str:
    out = processed_path("cifar10")
    if os.path.exists(out):
        return out
    import torch.nn.functional as F
    from torchvision import datasets as tvd
    from .models import vgg16_feature_extractor

    dev = device if torch.cuda.is_available() else "cpu"
    vgg = vgg16_feature_extractor().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed_half(imgs):
        x = F.interpolate(imgs.to(dev), size=(half_size, half_size),
                          mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return vgg(x).flatten(1).cpu()

    def to_arrays(train):
        ds = tvd.CIFAR10(RAW, train=train, download=True)
        x = torch.as_tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        y = np.array(ds.targets, dtype=np.int64)
        left, right = x[:, :, :, :16], x[:, :, :, 16:]
        ea, eb = [], []
        for i in range(0, len(x), batch):
            ea.append(embed_half(left[i:i + batch]))
            eb.append(embed_half(right[i:i + batch]))
        return torch.cat(ea).numpy(), torch.cat(eb).numpy(), y

    print(f"  cifar10: embedding halves with VGG16 (half_size={half_size}) on {dev} ...")
    xa_tr, xb_tr, y_tr = to_arrays(True)
    xa_te, xb_te, y_te = to_arrays(False)
    _save("cifar10",
          xa_train=xa_tr, xb_train=xb_tr, y_train=y_tr,
          xa_test=xa_te, xb_test=xb_te, y_test=y_te)
    print(f"  cifar10: A{xa_tr.shape[1]} B{xb_tr.shape[1]} train={len(y_tr)} test={len(y_te)}")
    return out


PREPARE = {
    "credit": prepare_credit,
    "vehicle": prepare_vehicle,
    "mnist": prepare_mnist,
    "nuswide": prepare_nuswide,
    "cifar10": prepare_cifar10,
}


def load(name: str, split: str = "train"):
    path = processed_path(name)
    if not os.path.exists(path):
        PREPARE[name]()
    d = torch.load(path, weights_only=True)
    return (d[f"xa_{split}"].float(), d[f"xb_{split}"].float(), d[f"y_{split}"].long())


def dims(name: str):
    xa, xb, _ = load(name, "train")
    return int(np.prod(xa.shape[1:])), int(np.prod(xb.shape[1:]))
