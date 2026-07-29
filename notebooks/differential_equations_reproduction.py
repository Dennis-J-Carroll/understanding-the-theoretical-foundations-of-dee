import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Are deep nets secretly solving differential equations?

        A tutorial walkthrough of a small reproduction of the foundational claim in
        **arXiv:2603.18331**, *"Understanding the Theoretical Foundations of Deep
        Neural Networks through Differential Equations."* That paper is a **survey**
        — it has no experiments of its own, so there's no number to match. Instead,
        we test the equation it's built on.

        **The claim (its Eq. 1-2):** a residual block

        ```
        h_{k+1} = h_k + f_theta(h_k)
        ```

        is exactly a **forward-Euler discretization** of the ODE `dh/dt = f_theta(h)`
        with step size `dt = 1`. Everything else in the survey — Neural ODEs, better
        numerical schemes as architectures, stability theory as an analysis tool —
        leans on this correspondence being real, not just a suggestive analogy.

        This notebook walks through what we did and shows the already-produced
        results. It does not require re-running any training — the numbers and
        figures below come from four short (CPU, <1 minute total) experiments run
        via `orx`; see the [full report](../reports/differential-equations-reproduction/report.md)
        for methodology detail.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The headline result

        If the correspondence is real, then a **fixed, trained** `f_theta`,
        re-simulated at finer step counts, should converge to *something* — a fixed
        underlying flow — rather than drift arbitrarily. Forward-Euler's error is
        known to shrink linearly in the step size `dt` (order 1).

        We trained a weight-tied residual block on a two-moons classification toy
        task at depth 8, then re-ran the *same trained network* at 16, 32, ..., 256
        steps and compared each output to a very fine reference (1024 steps).
        """
    )
    return


@app.cell
def _(mo):
    c1_dt = [0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    c1_diff = [0.15929502248764038, 0.10434455424547195, 0.06126927584409714,
               0.033100005239248276, 0.016412930563092232, 0.007248320616781712]
    c1_order = 0.8909496290263575

    mo.md(f"**Empirical convergence order: {c1_order:.2f}** (theoretical forward-Euler order: 1)")
    return c1_diff, c1_dt, c1_order


@app.cell(hide_code=True)
def _(c1_diff, c1_dt, mo, plt):
    import numpy as _np

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(c1_dt, c1_diff, "o-", color="#2166ac", linewidth=2, markersize=7, label="observed")
    _ref = _np.array(c1_dt)
    _scale = c1_diff[0] / _ref[0]
    _ax.loglog(_ref, _scale * _ref, "--", color="gray", linewidth=1.5, label="theoretical order 1")
    _ax.set_xlabel("step size  dt = 1/L")
    _ax.set_ylabel("mean |output diff| vs. reference (L=1024)")
    _ax.set_title("Refining the discretization converges the learned map")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Does a better integrator help? (C2: midpoint/RK2)

        The survey also claims that swapping in a higher-order numerical scheme
        (as PolyNet and FractalNet effectively do) should discretize *more
        accurately* at the same step budget. We swapped the Euler update for a
        midpoint (RK2) step — `h + dt*f(h + dt/2*f(h))` — and repeated the same test.
        """
    )
    return


@app.cell
def _(mo):
    c2_dt = [0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    c2_diff = [0.007992769591510296, 0.002667277352884412, 0.0004991462919861078,
               0.00013718723494093865, 3.5217195545556024e-05, 8.515891749993898e-06]
    c2_order = 1.9989645534226612

    mo.md(
        f"**Empirical convergence order: {c2_order:.2f}** (theoretical RK2 order: 2) "
        f"— roughly 20x lower error than Euler at 8 steps, 850x lower at 256 steps."
    )
    return c2_diff, c2_dt, c2_order


@app.cell(hide_code=True)
def _(c1_diff, c1_dt, c2_diff, c2_dt, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(c1_dt, c1_diff, "o-", color="#2166ac", linewidth=2, markersize=7, label="Euler (order 0.89)")
    _ax.loglog(c2_dt, c2_diff, "s-", color="#d6604d", linewidth=2, markersize=7, label="Midpoint/RK2 (order 2.00)")
    _ax.set_xlabel("step size  dt = 1/L")
    _ax.set_ylabel("mean |output diff| vs. reference")
    _ax.set_title("A higher-order scheme discretizes more accurately")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The continuous limit (C3: Neural ODE)

        The survey frames Neural ODEs as the limit a ResNet approaches as depth
        goes to infinity. We trained `f_theta` directly with an adaptive ODE solver
        (`torchdiffeq`, dopri5 — no discrete steps during training at all), then
        discretized *that* trained field with plain Euler at varying step counts.
        If the "continuous limit" framing is right, Euler's approximation should
        converge back to the adaptive solver's answer as steps increase.
        """
    )
    return


@app.cell
def _(mo):
    c3_dt = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
    c3_diff = [0.076602503657341, 0.04735415428876877, 0.027345644310116768,
               0.017647206783294678, 0.01077090110629797, 0.006072803400456905]
    c3_order = 0.7235917775829267

    mo.md(f"**Empirical convergence order: {c3_order:.2f}** — diffs shrink monotonically toward the adaptive solve.")
    return c3_diff, c3_dt


@app.cell(hide_code=True)
def _(c3_diff, c3_dt, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(c3_dt, c3_diff, "^-", color="#4daf4a", linewidth=2, markersize=8, label="Euler vs. adaptive solve")
    _ax.set_xlabel("step size  dt = T/L")
    _ax.set_ylabel("mean |output diff| vs. adaptive (dopri5) solution")
    _ax.set_title("Discrete Euler stacks converge to the continuous Neural ODE")
    _ax.legend()
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Negative control (C4): is this just "deep nets are smooth"?

        Everything above could, in principle, be an artifact of deep networks
        generically getting smoother as they get deeper — nothing to do with the
        residual/Euler structure specifically. To check, we removed the skip
        connection and weight-tying entirely: independent weights per layer, no
        `+`, no `dt`. Same depth sweep (8 to 128 layers).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, plt):
    _c1_depths = [8, 16, 32, 64, 128, 256]
    _c1_acc = [0.9416666626930237, 0.9083333611488342, 0.8833333253860474,
               0.8666666746139526, 0.8583333492279053, 0.8500000238418579]
    _c4_depths = [8, 16, 32, 64, 128]
    _c4_acc = [0.49166667461395264] * 5

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.semilogx(_c1_depths, _c1_acc, "o-", color="#2166ac", linewidth=2, markersize=7, label="C1: residual/Euler")
    _ax.semilogx(_c4_depths, _c4_acc, "x--", color="#999999", linewidth=2, markersize=9, label="C4: plain stack")
    _ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.6)
    _ax.set_xlabel("depth (layers / steps L)")
    _ax.set_ylabel("test accuracy")
    _ax.set_ylim(0.4, 1.0)
    _ax.set_title("Plain deep stacks collapse to chance; the residual construction doesn't")
    _ax.legend(loc="lower left")
    _ax.grid(True, which="both", alpha=0.3)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Every depth from 8 to 128 collapsed to **49.17% test accuracy** — chance
        level on this balanced task — with near-machine-precision agreement between
        depths (the network isn't converging to a flow, it's just failing to train:
        the vanishing-gradient problem that historically motivated ResNets in the
        first place).

        ## Assessment summary

        | Claim | Result | Assessment |
        |---|---|---|
        | C1: ResNet block = Euler step | order 0.89 (theory 1) | aligned |
        | C2: better scheme = more accurate | order 2.00 (theory 2) | aligned |
        | C3: Neural ODE = continuous limit | order 0.72, smooth convergence | aligned |
        | C4: negative control | collapses to chance at all depths | supports the distinction |

        This was a **toy-scale, single-seed, CPU-only** check (two-moons, 8-dim
        hidden state) — see the [full report](../reports/differential-equations-reproduction/report.md)
        for what a fuller reproduction (real image benchmarks, multiple seeds) would
        still need to establish.

        ---

        ### Running this notebook yourself

        This notebook is self-contained — the numbers/figures above are embedded
        directly, no re-run required. To explore or edit it locally:

        ```sh
        marimo edit notebooks/differential_equations_reproduction.py   # interactive editing
        marimo run notebooks/differential_equations_reproduction.py    # read-only app view
        ```
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    return mo, plt


if __name__ == "__main__":
    app.run()
