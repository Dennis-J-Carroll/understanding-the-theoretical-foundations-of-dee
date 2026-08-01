# understanding-the-theoretical-foundations-of-dee

## Reproduction: arXiv:2506.09985 — "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/blob/main/notebooks/vjepa2_temporal_order_reproduction.py)

V-JEPA 2 (Meta FAIR) claims that self-supervised **video** pretraining gives
frozen features genuine motion/temporal sensitivity that image pretraining
doesn't — evidenced by a large probe-accuracy gap over image-style encoders on
a motion-centric benchmark (SSv2: 77.3% vs. 55.4–69.7%) and a much smaller gap
on an appearance-centric one (ImageNet: 84.6%, close to peers).

- **What was done**: downscaled the claim to a real-vs-shuffled frame-order
  probe on frozen features — same 64 frames either way, only the order
  permuted — using the real released `facebook/vjepa2-vitl-fpc64-256`
  checkpoint (~300M params) vs. a `facebook/dinov2-small` image-encoder
  baseline (frames encoded independently, mean-pooled: provably
  order-invariant). Data: `nateraw/kinetics-mini` (public, no license gate).
- **Verdict**: **Reproduced (directional proxy, small-N)**. V-JEPA 2: 95.0%
  held-out probe accuracy at telling real order from shuffled. DINOv2
  baseline: exactly 50.0% (chance), with a measured real-vs-shuffled feature
  difference of 9.5×10⁻⁷ — float32 noise, confirming the order-invariance
  argument empirically rather than by assumption.
- **Paper vs. reproduced**: paper's SSv2 probe gap (77.3% vs. 55.4–69.7%,
  ViT-g/1B, full val set, gated data) vs. this reproduction's order-detection
  proxy gap (95.0% vs. 50.0%, ViT-L/300M, N=20 clips, public data) — same
  directional claim, different task/scale/data; not a claim of matching 77.3%.
- **Compute**: `local` backend (CPU), ~83 min wall-clock, no cloud spend.
- **Links**: [molab tutorial notebook](https://molab.marimo.io/github/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/blob/main/notebooks/vjepa2_temporal_order_reproduction.py) · [full report](reports/vjepa2-temporal-order-sensitivity/report.md)

### Experiment log

| Branch | Purpose / change | Run command | Verdict | Compute |
|---|---|---|---|---|
| `main` | Publication surface (README, report, notebook) | Not run as an experiment (publication surface) | — | — |
| [`orx/v1-v-jepa-2-frozen-feature-temporal-order-sensit`](https://github.com/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/blob/orx/v1-v-jepa-2-frozen-feature-temporal-order-sensit/run_probe.py) | Root: frozen V-JEPA 2 + DINOv2 feature extraction, real/shuffled frame-order pairing, linear probes | `uv venv --clear .venv && . .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install "transformers>=4.53" huggingface_hub av pillow scikit-learn numpy && python run_probe.py` | Reproduced (directional proxy, small-N) — V-JEPA 2 95.0% vs. baseline 50.0% held-out probe accuracy | local CPU, ~83 min |

---

## Reproduction: arXiv:2603.18331 — "Understanding the Theoretical Foundations of Deep Neural Networks through Differential Equations"

This paper is a **survey** with no original experiment table of its own — every result it cites is from other papers. So instead of matching a reported number, we tested the survey's foundational claim itself: **a residual block `h_{k+1} = h_k + f_theta(h_k)` is a forward-Euler discretization of an ODE** (its Eq. 1–2), and checked whether the consequences of that claim actually hold empirically.

- **What was done**: trained a weight-tied residual (Euler) block on a two-moons classification toy task, then re-simulated the *same trained network* at finer step counts and measured whether its output converges (as a numerical integrator would) — plus three follow-ons: a higher-order (midpoint/RK2) discretization, a Neural ODE (`torchdiffeq` adaptive solve) as the continuous limit, a plain non-residual stack as a negative control, and a single-factor ablation to find out which missing ingredient the negative control's collapse was actually due to.
- **Compute**: local CPU (4-core laptop, no GPU) — `--backend local`, all five runs together take well under a minute of wall-clock training.
- **Downscaling**: 2D two-moons toy task, 8-dim hidden state, 400 training epochs, single seed — deliberately tiny, since the claim under test is architectural and doesn't require scale to falsify.
- **Assessment**: **aligned** on all three positive claims (C1 order 0.89 vs. theoretical 1; C2 order 2.00 vs. theoretical 2; C3 order 0.72, smooth convergence to the adaptive solution). The negative control (C4) collapsed to chance accuracy at every depth, but it changes three variables (skip, weight-tying, `dt`) at once, so alone it's **inconclusive** about which one is responsible. A follow-up ablation (C5, skip connection restored only) trains at C1-level accuracy at every depth — isolating the skip connection, not weight-tying, as the necessary ingredient.
- **Paper vs. observed**: the paper states an analytical equivalence, not a number — there is nothing to numerically match. The comparison here is "does the stated equivalence have the empirical consequences it should," and it does, at this scale.

Full write-up with figures: [`reports/differential-equations-reproduction/report.md`](reports/differential-equations-reproduction/report.md).
Tutorial notebook (opens with the evidence, no rerun required): [`notebooks/differential_equations_reproduction.py`](notebooks/differential_equations_reproduction.py).

### Experiment log

| Branch | Purpose / change | Run command | Assessment | Compute |
|---|---|---|---|---|
| `main` | Publication surface (README, report, notebook) | Not run as an experiment (publication surface) | — | — |
| [`orx/c1-resnet-as-forward-euler-discretization-2`](https://github.com/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/tree/orx/c1-resnet-as-forward-euler-discretization-2) | Baseline (C1): weight-tied residual/Euler block, refine step count on the *same trained* network vs. an L=1024 reference | `uv venv --clear .venv && . .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install numpy scikit-learn torchdiffeq && python train.py` | Aligned — empirical convergence order 0.89 (theory: 1) | local CPU |
| [`orx/c2-midpoint-rk2-discretization-vs-euler`](https://github.com/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/tree/orx/c2-midpoint-rk2-discretization-vs-euler) | Child of C1: swap Euler block for midpoint/RK2 block, same pipeline | `uv venv --clear .venv && . .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install numpy scikit-learn torchdiffeq && python train.py` | Aligned — empirical order 2.00 (theory: 2), 20–850x lower error than C1 at matched step counts | local CPU |
| [`orx/c3-neural-ode-adaptive-solve-vs-euler-discretiza`](https://github.com/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/tree/orx/c3-neural-ode-adaptive-solve-vs-euler-discretiza) | Child of C1: train via `torchdiffeq` adjoint (dopri5) instead of fixed Euler steps; compare Euler-discretized approximation of the trained field to the adaptive solution | `uv venv --clear .venv && . .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install numpy scikit-learn torchdiffeq && python train.py` | Aligned — empirical order 0.72, diffs shrink monotonically toward the adaptive solve | local CPU |
| [`orx/c4-plain-non-residual-stack-negative-control`](https://github.com/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/tree/orx/c4-plain-non-residual-stack-negative-control) | Child of C1: negative control — independent per-layer weights, no skip connection, no `dt` scaling (changes 3 factors at once) | `uv venv --clear .venv && . .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install numpy scikit-learn torchdiffeq && python train.py` | Inconclusive alone — collapses to chance accuracy (49.17%) at every depth 8–128, but can't attribute the cause to a single factor; see C5 | local CPU |
| [`orx/c5-skip-connection-only-ablation-untied-weights`](https://github.com/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/tree/orx/c5-skip-connection-only-ablation-untied-weights) | Child of C1: ablation — same as C4 but with the skip connection restored (still untied, still no `dt`) | `uv venv --clear .venv && . .venv/bin/activate && uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install numpy scikit-learn torchdiffeq && python train.py` | Isolates the cause — trains at C1-level accuracy (0.87–0.95) at every depth; skip connection, not weight-tying, prevents C4's collapse | local CPU |

Note: an earlier baseline attempt (superseded, not listed above) failed at dependency install because `uv pip install torch --index-url <cpu-url> numpy scikit-learn torchdiffeq` on one line silently drops the non-torch packages from the default PyPI index — fixed by splitting the torch-CPU install from the rest.
