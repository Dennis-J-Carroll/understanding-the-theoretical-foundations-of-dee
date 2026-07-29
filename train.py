import json
import math

import torch
import torch.nn as nn

from data import get_moons_data
from model import NeuralODEClassifier, euler_integrate

SEED = 0
EPOCHS = 400
LR = 0.05
T = 1.0
REFINE_STEPS = [4, 8, 16, 32, 64, 128]

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
    t_span = torch.tensor([0.0, T])

    model = NeuralODEClassifier(T=T)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        opt.zero_grad()
        logits = model(Xtr, t_span)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"epoch={epoch} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        adaptive_logits = model(Xte, t_span)
        adaptive_acc = accuracy(adaptive_logits, yte)
        adaptive_probs = torch.sigmoid(adaptive_logits)
        print(f"adaptive_solver_test_acc={adaptive_acc:.4f}")

        h0 = model.embed(Xte)
        results = []
        for L in REFINE_STEPS:
            hT = euler_integrate(model.block.f, h0, num_steps=L, T=T)
            logits_L = model.head(hT).squeeze(-1)
            probs_L = torch.sigmoid(logits_L)
            diff = (probs_L - adaptive_probs).abs().mean().item()
            acc = accuracy(logits_L, yte)
            row = {"num_steps": L, "dt": T / L, "test_acc": acc, "mean_abs_diff_to_adaptive_solution": diff}
            results.append(row)
            print(json.dumps(row))

    order = fit_convergence_order(results, "mean_abs_diff_to_adaptive_solution")

    summary = {
        "variant": "C3_neural_ode_adaptive",
        "adaptive_solver_test_acc": adaptive_acc,
        "empirical_euler_convergence_order_to_adaptive_solution": order,
        "refinement_table": results,
    }
    print("SUMMARY_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
