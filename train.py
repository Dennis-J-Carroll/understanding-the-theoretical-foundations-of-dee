import json
import math

import torch
import torch.nn as nn

from data import get_moons_data
from model import ODEClassifier, EulerODEBlock

SEED = 0
TRAIN_STEPS = 8            # depth (Euler steps) used during training
EPOCHS = 400
LR = 0.05
REFINE_STEPS = [8, 16, 32, 64, 128, 256]
REFERENCE_STEPS = 1024     # proxy for the continuous flow

torch.manual_seed(SEED)


def accuracy(logits, y):
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == y).float().mean().item()


def fit_convergence_order(results, diff_key):
    xs = [math.log(r["dt"]) for r in results if r[diff_key] > 0]
    ys = [math.log(r[diff_key]) for r in results if r[diff_key] > 0]
    if len(xs) < 2:
        return float("nan")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    return cov / var if var > 0 else float("nan")


def main():
    Xtr, ytr, Xte, yte = get_moons_data(seed=SEED)

    model = ODEClassifier(block_cls=EulerODEBlock)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        opt.zero_grad()
        logits = model(Xtr, TRAIN_STEPS)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"epoch={epoch} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_depth_acc = accuracy(model(Xte, TRAIN_STEPS), yte)
        print(f"train_depth={TRAIN_STEPS} test_acc={train_depth_acc:.4f}")

        ref_probs = torch.sigmoid(model(Xte, REFERENCE_STEPS))

        results = []
        for L in REFINE_STEPS:
            logits_L = model(Xte, L)
            probs_L = torch.sigmoid(logits_L)
            diff = (probs_L - ref_probs).abs().mean().item()
            acc = accuracy(logits_L, yte)
            row = {"num_steps": L, "dt": 1.0 / L, "test_acc": acc, "mean_abs_diff_to_reference": diff}
            results.append(row)
            print(json.dumps(row))

    order = fit_convergence_order(results, "mean_abs_diff_to_reference")

    summary = {
        "variant": "C1_euler_weight_tied",
        "train_depth": TRAIN_STEPS,
        "test_acc_at_train_depth": train_depth_acc,
        "empirical_convergence_order": order,
        "refinement_table": results,
    }
    print("SUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
