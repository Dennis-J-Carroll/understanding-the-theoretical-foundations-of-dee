import json

import torch
import torch.nn as nn

from data import get_moons_data
from model import SkipOnlyClassifier

SEED = 0
EPOCHS = 400
LR = 0.05
DEPTHS = [8, 16, 32, 64, 128]


def accuracy(logits, y):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == y).float().mean().item()


def train_one(depth, Xtr, ytr, Xte, yte, seed):
    torch.manual_seed(seed)
    model = SkipOnlyClassifier(depth=depth)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        test_logits = model(Xte)
        acc = accuracy(test_logits, yte)
        probs = torch.sigmoid(test_logits)
    return acc, probs


def main():
    Xtr, ytr, Xte, yte = get_moons_data(seed=SEED)

    accs = {}
    probs_by_depth = {}
    for depth in DEPTHS:
        acc, probs = train_one(depth, Xtr, ytr, Xte, yte, seed=SEED)
        accs[depth] = acc
        probs_by_depth[depth] = probs
        print(f"depth={depth} test_acc={acc:.4f}")

    results = []
    for d1, d2 in zip(DEPTHS[:-1], DEPTHS[1:]):
        diff = (probs_by_depth[d2] - probs_by_depth[d1]).abs().mean().item()
        row = {"depth_pair": [d1, d2], "mean_abs_diff": diff}
        results.append(row)
        print(json.dumps(row))

    summary = {
        "variant": "C5_skip_only_ablation",
        "depths": DEPTHS,
        "test_acc_by_depth": accs,
        "consecutive_depth_diffs": results,
    }
    print("SUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
