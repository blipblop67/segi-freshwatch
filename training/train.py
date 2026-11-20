"""Fine-tune a two-head MobileNetV3-Large: produce type + freshness.

Runs on the Jetson itself (CUDA + AMP) or any CUDA workstation.

    python training/train.py --data data/processed --out models/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
FRESHNESS = ["fresh", "rotten"]


class FreshnessNet(nn.Module):
    def __init__(self, n_produce: int, pretrained: bool = True) -> None:
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.mobilenet_v3_large(weights=w)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(960, 1280), nn.Hardswish(), nn.Dropout(0.2))
        self.produce_head = nn.Linear(1280, n_produce)
        self.freshness_head = nn.Linear(1280, len(FRESHNESS))

    def forward(self, x):
        z = self.neck(torch.flatten(self.pool(self.features(x)), 1))
        return self.produce_head(z), self.freshness_head(z)


class ManifestDataset(Dataset):
    def __init__(self, root: Path, split: str, produce: list[str], tf) -> None:
        with open(root / "manifest.csv") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.items = [(root / r["path"], produce.index(r["produce"]), FRESHNESS.index(r["freshness"])) for r in rows]
        self.tf = tf

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, p, f = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), p, f


def build_transforms(size: int):
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.7, 1.4)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        # Hue jitter kept tiny: colour shift *is* the spoilage signal.
        transforms.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.15, hue=0.015),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # Plain stretch-resize: identical to the CUDA zone preprocessing in production.
    evaluate = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                                   transforms.Normalize(MEAN, STD)])
    return train, evaluate


@torch.no_grad()
def evaluate(model, loader, device, produce):
    model.eval()
    n = p_ok = f_ok = 0
    conf = [[0, 0], [0, 0]]
    per_produce = defaultdict(lambda: [0, 0])
    for x, p, f in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lp, lf = model(x)
        pp, pf = lp.argmax(1).cpu(), lf.argmax(1).cpu()
        n += len(p)
        p_ok += (pp == p).sum().item()
        f_ok += (pf == f).sum().item()
        for t, q, k in zip(f.tolist(), pf.tolist(), p.tolist(), strict=False):
            conf[t][q] += 1
            per_produce[produce[k]][0] += int(t == q)
            per_produce[produce[k]][1] += 1
    rotten_recall = conf[1][1] / max(conf[1][0] + conf[1][1], 1)
    fresh_recall = conf[0][0] / max(conf[0][0] + conf[0][1], 1)
    return {
        "n": n,
        "produce_acc": p_ok / max(n, 1),
        "freshness_acc": f_ok / max(n, 1),
        "rotten_recall": rotten_recall,
        "fresh_recall": fresh_recall,
        "confusion_fresh_rotten": conf,
        "freshness_acc_by_produce": {k: v[0] / v[1] for k, v in sorted(per_produce.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="models")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(root / "manifest.csv") as f:
        produce = sorted({r["produce"] for r in csv.DictReader(f)})
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    ttf, etf = build_transforms(a.size)
    def dl(split, tf, shuffle):
        return DataLoader(
            ManifestDataset(root, split, produce, tf), batch_size=a.batch, shuffle=shuffle, num_workers=a.workers,
            pin_memory=True, drop_last=shuffle, persistent_workers=True)
    train_dl, val_dl, test_dl = dl("train", ttf, True), dl("val", etf, False), dl("test", etf, False)
    print(f"train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)} produce={produce}")

    model = FreshnessNet(len(produce)).to(device).to(memory_format=torch.channels_last)
    params = [{"params": model.features.parameters(), "lr": a.lr * 0.3},
              {"params": [*model.neck.parameters(), *model.produce_head.parameters(),
                          *model.freshness_head.parameters()], "lr": a.lr}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], epochs=a.epochs,
                                                steps_per_epoch=len(train_dl), pct_start=0.15)
    scaler = torch.amp.GradScaler()
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_path = -1.0, out / "freshness_best.pt"
    for epoch in range(a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for x, p, f in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            p, f = p.to(device), f.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                lp, lf = model(x)
                loss = 0.5 * ce(lp, p) + 1.0 * ce(lf, f)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        m = evaluate(model, val_dl, device, produce)
        print(f"epoch {epoch + 1}/{a.epochs} loss={run / len(train_dl):.4f} "
              f"val_fresh={m['freshness_acc']:.4f} val_rotten_recall={m['rotten_recall']:.4f} "
              f"val_produce={m['produce_acc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        score = m["freshness_acc"] + 0.5 * m["rotten_recall"]
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "produce": produce}, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    test = evaluate(model, test_dl, device, produce)
    print("TEST", json.dumps(test, indent=2))
    labels = {"produce": produce, "freshness": FRESHNESS, "input_size": a.size, "mean": MEAN, "std": STD,
              "arch": "mobilenet_v3_large_2head", "dataset": "Project-AgML/fresh_rotten_fruit_classification (raw)"}
    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    (out / "metrics.json").write_text(json.dumps({"test": test, "epochs": a.epochs}, indent=2))


if __name__ == "__main__":
    main()
