# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "pillow",
#     "transformers>=4.53",
#     "huggingface_hub",
#     "av",
# ]
# ///
import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Does a video-pretrained model actually "see" motion?

        Most vision encoders are trained on still images: they learn what things
        *look like*. **V-JEPA 2** (Meta FAIR, [arXiv:2506.09985](https://arxiv.org/abs/2506.09985))
        is trained on over a million hours of *video* instead, predicting masked
        chunks of a clip in representation space rather than pixel space. The
        paper's headline claim is that this makes its frozen features carry
        genuine **motion / temporal information** — not just "a picture, repeated" —
        in a way image-pretrained encoders structurally cannot.

        This notebook reproduces that claim at small scale and explains, with the
        actual math, *why* the two kinds of encoders must behave differently.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The paper's result

        V-JEPA 2 evaluates frozen features with a lightweight probe on six
        classification benchmarks. The gap is largest on the **motion-centric**
        one and much smaller on the **appearance-centric** one:

        | Benchmark | What it needs | V-JEPA 2 ViT-g | Best other frozen encoder |
        |---|---|---|---|
        | Something-Something v2 (motion) | telling *what happened*, not just *what's in frame* | **77.3%** | InternVideo2-1B: 69.7%, PEcoreG: 55.4% |
        | ImageNet (appearance) | recognizing *what's in frame* | 84.6% | close to peers |

        Same model, same evaluation protocol — but the advantage over image-style
        encoders is far larger on the task that actually requires understanding
        *order and change over time*. That gap is the claim we're testing: does
        video pretraining specifically buy you order-sensitivity that image
        pretraining doesn't?

        SSv2 and the paper's 1B-parameter ViT-g are gated / expensive, so this
        reproduction substitutes a public model, a public dataset, and a proxy
        task that isolates the same mechanism directly — see
        [Limitations & provenance](#limitations-provenance) below for the exact
        substitutions.
        """
    )
    return


@app.cell
def _():
    # Embedded verbatim from the frozen orx run's RESULT_SUMMARY_JSON.
    # (see "Limitations & provenance" for the branch this came from)
    vjepa_train_acc = 1.000
    vjepa_test_acc = 0.950
    baseline_train_acc = 0.500
    baseline_test_acc = 0.500
    baseline_max_abs_diff = 9.5367431640625e-07
    vjepa_mean_cosine_dist = 0.08634673058986664
    n_clips_train = 10
    n_clips_test = 10
    n_probe_train = 20
    n_probe_test = 20
    total_wall_time_s = 4975.9
    return (
        baseline_max_abs_diff,
        baseline_test_acc,
        baseline_train_acc,
        n_clips_test,
        n_clips_train,
        n_probe_test,
        n_probe_train,
        total_wall_time_s,
        vjepa_mean_cosine_dist,
        vjepa_test_acc,
        vjepa_train_acc,
    )


@app.cell(hide_code=True)
def _(
    baseline_max_abs_diff,
    baseline_test_acc,
    baseline_train_acc,
    mo,
    n_clips_test,
    n_clips_train,
    n_probe_test,
    n_probe_train,
    total_wall_time_s,
    vjepa_mean_cosine_dist,
    vjepa_test_acc,
    vjepa_train_acc,
):
    mo.md(
        f"""
        ## Reproduction result

        > ### Verdict: **Reproduced** (directional proxy, small-N)

        Proxy task in place of SSv2 action-labels: freeze each encoder, show it
        the **same 64 frames** of a clip twice — once in the real order, once
        with those exact frames randomly permuted — and train a linear probe to
        tell real from shuffled apart from the pooled features alone.

        | | V-JEPA 2 (video-pretrained) | DINOv2 baseline (image-pretrained, mean-pooled) |
        |---|---|---|
        | Probe train accuracy | {vjepa_train_acc:.0%} | {baseline_train_acc:.0%} |
        | **Probe test accuracy (held-out clips)** | **{vjepa_test_acc:.0%}** | **{baseline_test_acc:.0%}** |
        | Real-vs-shuffled feature distance | mean cosine distance {vjepa_mean_cosine_dist:.3f} | max\\|diff\\| = {baseline_max_abs_diff:.1e} |

        **N** = {n_clips_train} train clips / {n_clips_test} held-out test clips
        (disjoint videos, 2 per class across 5 action classes) &rarr;
        {n_probe_train} probe-train / {n_probe_test} probe-test examples (each
        clip contributes one real-order and one shuffled-order example).
        Total compute: **{total_wall_time_s/60:.0f} min** CPU wall-clock (local
        backend, no cloud spend).
        """
    )
    return


@app.cell(hide_code=True)
def _(plt, vjepa_train_acc, vjepa_test_acc, baseline_train_acc, baseline_test_acc, mo):
    _labels = ["train", "test\n(held-out clips)"]
    _vjepa_vals = [vjepa_train_acc, vjepa_test_acc]
    _base_vals = [baseline_train_acc, baseline_test_acc]
    _x = [0, 1]
    _w = 0.32

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.bar([i - _w / 2 for i in _x], _vjepa_vals, _w, label="V-JEPA 2 (video-pretrained)", color="#2166ac")
    _ax.bar([i + _w / 2 for i in _x], _base_vals, _w, label="DINOv2 (image-pretrained, mean-pooled)", color="#b2182b")
    _ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="chance (50%)")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_labels)
    _ax.set_ylim(0, 1.08)
    _ax.set_ylabel("real-vs-shuffled probe accuracy")
    _ax.set_title("Can a linear probe tell real frame order from shuffled order?")
    _ax.legend(loc="lower center", framealpha=0.9)
    _ax.grid(axis="y", alpha=0.25)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Reading the chart:** each bar pair is one encoder's probe accuracy at
        telling real-order from shuffled-order clips apart, using only its frozen
        pooled features. Blue = V-JEPA 2, red = DINOv2, the dashed line is chance
        (50%, since the task is a balanced binary classification). The right pair
        (test set) is the number that matters — it's evaluated on clips the probe
        never saw. V-JEPA 2 clears chance by a wide margin; DINOv2 sits *exactly*
        on it, both train and test — not "close to," but numerically pinned.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, baseline_max_abs_diff):
    mo.md(
        f"""
        ## Robustness & interpretation

        ### Why DINOv2 sitting at exactly 50% isn't a coincidence

        DINOv2 encodes each of the 64 frames **independently** — no frame ever
        sees another frame during encoding — and the pooled feature is the mean
        over that set:

        ```
        pooled = mean(encode(frame_1), encode(frame_2), ..., encode(frame_64))
        ```

        A mean is a sum over a *set*, divided by its size. Permuting the order in
        which you visit a set before summing it cannot change the sum — order
        isn't even an input to that computation. So DINOv2's real-clip feature
        and shuffled-clip feature must be **identical**, not just similar, for
        *any* permutation, as a matter of arithmetic rather than learned behavior.
        This is checked directly in the run, not assumed: the largest absolute
        difference between any real/shuffled feature pair was
        **{baseline_max_abs_diff:.1e}** — float32 rounding noise, i.e. genuinely
        zero. A linear probe asked to separate identical points with opposite
        labels cannot beat chance on those points *even on the training set it
        memorizes* — which is exactly what the 50.0% training accuracy above is:
        not "the probe failed to find a pattern," but "there is provably no
        pattern to find."

        ### Why V-JEPA 2 can differ

        V-JEPA 2 encodes all 64 frames **jointly**, through self-attention where
        every space-time patch attends to every other patch, with positional
        structure (3D-RoPE) telling the model *where in time* each patch sits.
        Permuting frame order changes which content sits at which position, which
        changes what every patch attends to — so there is no algebraic reason for
        the pooled output to be order-invariant. Whether it *actually* is
        order-sensitive is an empirical question, which is what the probe above
        measures: 95% held-out accuracy, and a real mean cosine distance
        (0.086) between real and shuffled pooled features, not floating-point noise.

        ### What would falsify this

        If V-JEPA 2's held-out accuracy had also landed near 50% and its
        real/shuffled cosine distance had also been ~1e-6, that would falsify the
        claim at this scale — it would mean pretraining on video didn't actually
        buy order-sensitivity beyond what pooling independent frames gives you
        for free. That didn't happen.

        ### Honest limits of this evidence

        - **N is tiny.** 10 held-out clips (20 probe-test examples) is enough to
          see a large, clean effect, but not enough for a tight confidence
          interval — one flipped clip changes the test accuracy by 5 points.
        - **Single seed, single permutation per clip.** Different random shuffles
          could land on an easier or harder instance by chance.
        - **This is a proxy, not the paper's own benchmark.** Order-detection on
          5 Kinetics-mini action classes is a different task from SSv2 action
          classification on ~174 fine-grained classes, and probes a related but
          not identical capability.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Limitations & provenance {#limitations-provenance}

        ### Substitutions made for tractability

        | Paper (arXiv:2506.09985, Sec 5.1) | This reproduction | Why |
        |---|---|---|
        | V-JEPA 2 ViT-g, ~1B params | V-JEPA 2 **ViT-L**, ~300M params (`facebook/vjepa2-vitl-fpc64-256`) | Smallest released checkpoint; still the real architecture and real pretrained weights, not a toy |
        | Something-Something v2 / Jester (registration-gated) | **nateraw/kinetics-mini** (public, no license gate; also the dataset used in V-JEPA 2's own HF model-card example) | SSv2/Jester require a signed license agreement |
        | Action classification, ~174/27 classes, full val sets | **Real-vs-shuffled frame-order probe**, N=20 clips, 5 classes | Isolates the same order-sensitivity mechanism directly, at a scale a CPU run can finish |
        | Attentive probe, large eval sets | Linear probe (`sklearn.LogisticRegression`), 20 held-out examples | Proportionate to the small-N setting; directional, not a tight estimate |

        Not attempted here: SSv2/Jester probe accuracy at the paper's own scale,
        VidQA (needs an 8B-parameter LLM aligner), robot planning (needs a
        Franka arm and the Droid dataset), and action anticipation on
        Epic-Kitchens-100.

        ### Provenance

        | Branch | Contains | Verdict | Compute |
        |---|---|---|---|
        | [`orx/v1-v-jepa-2-frozen-feature-temporal-order-sensit`](https://github.com/Dennis-J-Carroll/understanding-the-theoretical-foundations-of-dee/blob/orx/v1-v-jepa-2-frozen-feature-temporal-order-sensit/run_probe.py) | Runner (`run_probe.py`): frozen V-JEPA 2 + DINOv2 feature extraction, real/shuffled pairing, linear probes, fixed seed | Reproduced (directional, small-N) | ~83 min, CPU, local backend |

        The run command is a fixed contract on that branch — see the repository
        README's provenance table for the exact command used, copied verbatim
        from `orx exp status`. Only the branch above produced numbers; two
        earlier attempts on the same node failed on environment setup
        (a missing `torchvision` backend for HF's image/video processors, worked
        around by re-implementing each processor's documented preprocessing
        directly in NumPy/PIL) and are part of that same branch's history, not
        separate experiments.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Optional: interactive GPU lab

        Everything above is frozen, already-run evidence — nothing below is
        needed to read the result. This section is a **live, illustrative
        demo** of the *mechanism*, not additional reproduction evidence: it
        downloads a couple of public video clips fresh and runs real V-JEPA 2 /
        DINOv2 forward passes, so you can watch the effect above appear
        gradually rather than as one fixed permutation.

        **What you control:** how many demo clips to use, then press the button
        to sweep a **partial-shuffle strength** (0% = untouched, 100% = fully
        shuffled, matching the reproduction above) and plot how far each
        encoder's pooled features move from their unshuffled starting point.

        Runs real inference on molab's attached GPU — nothing starts until you
        click the button below.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    try:
        import torch as torch_module

        cuda_available = torch_module.cuda.is_available()
        device_name = torch_module.cuda.get_device_name(0) if cuda_available else "CPU"
        torch_ok = True
    except ImportError:
        torch_module = None
        cuda_available = False
        device_name = "torch not installed"
        torch_ok = False

    mo.md(
        f"**Device check:** CUDA available = `{cuda_available}` &mdash; running on **{device_name}**"
        if torch_ok
        else "*torch is not installed in this environment* &mdash; the lab below is "
        "unavailable here but will work when this notebook is opened in molab."
    )
    return cuda_available, torch_module, torch_ok


@app.cell(hide_code=True)
def _(torch_ok):
    try:
        from transformers import AutoModel
        from huggingface_hub import hf_hub_download
        import av as av_module

        lab_ready = torch_ok
        lab_import_error = None
    except ImportError as _e:
        AutoModel = None
        hf_hub_download = None
        av_module = None
        lab_ready = False
        lab_import_error = str(_e)
    return AutoModel, av_module, hf_hub_download, lab_import_error, lab_ready


@app.cell(hide_code=True)
def _(np):
    from PIL import Image

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _resize_center_crop(img, short_edge, crop, resample):
        w, h = img.size
        if w <= h:
            new_w, new_h = short_edge, round(h * short_edge / w)
        else:
            new_h, new_w = short_edge, round(w * short_edge / h)
        img = img.resize((new_w, new_h), resample)
        left, top = (new_w - crop) // 2, (new_h - crop) // 2
        return img.crop((left, top, left + crop, top + crop))

    def preprocess_for_vjepa(frame_hwc_uint8):
        img = Image.fromarray(frame_hwc_uint8)
        img = _resize_center_crop(img, 292, 256, Image.Resampling.BILINEAR)
        arr = (np.asarray(img).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)

    def preprocess_for_dino(frame_hwc_uint8):
        img = Image.fromarray(frame_hwc_uint8)
        img = _resize_center_crop(img, 256, 224, Image.Resampling.BICUBIC)
        arr = (np.asarray(img).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return arr.transpose(2, 0, 1)

    return preprocess_for_dino, preprocess_for_vjepa


@app.cell(hide_code=True)
def _(lab_ready, mo):
    num_clips_slider = mo.ui.slider(1, 5, value=3, step=1, label="Number of demo clips")
    run_lab_button = mo.ui.run_button(
        label="Run GPU lab (downloads clips + V-JEPA 2 / DINOv2 forward passes)",
        disabled=not lab_ready,
    )
    mo.hstack([num_clips_slider, run_lab_button])
    return num_clips_slider, run_lab_button


@app.cell
def _(
    AutoModel,
    av_module,
    hf_hub_download,
    lab_import_error,
    lab_ready,
    mo,
    np,
    num_clips_slider,
    preprocess_for_dino,
    preprocess_for_vjepa,
    run_lab_button,
    torch_module,
):
    mo.stop(
        not lab_ready,
        mo.md(
            f"*Lab unavailable in this environment ({lab_import_error or 'torch not installed'}). "
            f"Open this notebook in molab to run it.*"
        ),
    )
    mo.stop(not run_lab_button.value, mo.md("Click **Run GPU lab** above to start."))

    import time as _time_mod

    _t0 = _time_mod.time()
    _device = "cuda" if torch_module.cuda.is_available() else "cpu"

    _demo_pool = [
        ("marching", "5EVgOrjJjuM_000160_000170.mp4"),
        ("marching", "5SjV5j8f6rw_000009_000019.mp4"),
        ("bowling", "-jOClYqKtE8_000003_000013.mp4"),
        ("bowling", "1W7HNDBA4pA_000002_000012.mp4"),
        ("archery", "0S-P4lr_c7s_000022_000032.mp4"),
    ][: num_clips_slider.value]
    _shuffle_levels = [0, 20, 40, 60, 80, 100]

    _vjepa = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256").to(_device).eval()
    _dino = AutoModel.from_pretrained("facebook/dinov2-small").to(_device).eval()

    def _decode_64(path):
        _container = av_module.open(path)
        _frames = [f.to_ndarray(format="rgb24") for f in _container.decode(video=0)]
        _container.close()
        _frames = np.stack(_frames, axis=0)
        _t = _frames.shape[0]
        _idx = np.clip(np.linspace(0, _t - 1, 64).round().astype(int), 0, _t - 1)
        return _frames[_idx]

    def _partial_shuffle(frames_64, pct, seed):
        _rng = np.random.RandomState(seed)
        _k = round(pct / 100 * 64)
        _out = frames_64.copy()
        if _k >= 2:
            _positions = _rng.choice(64, size=_k, replace=False)
            _permuted = _rng.permutation(_positions)
            _out[_positions] = frames_64[_permuted]
        return _out

    def _vjepa_feat(frames_64):
        _chw = [preprocess_for_vjepa(f) for f in frames_64]
        _video = torch_module.from_numpy(np.stack(_chw, axis=0)).float().unsqueeze(0).to(_device)
        with torch_module.no_grad():
            _feats = _vjepa.get_vision_features(pixel_values_videos=_video)
        return _feats.mean(dim=1).squeeze(0).cpu().numpy()

    def _dino_feat(frames_64):
        _chw = [preprocess_for_dino(f) for f in frames_64]
        _pv = torch_module.from_numpy(np.stack(_chw, axis=0)).float().to(_device)
        with torch_module.no_grad():
            _out = _dino(pixel_values=_pv).last_hidden_state
        return _out.mean(dim=(0, 1)).cpu().numpy()

    _vjepa_curve = {lvl: [] for lvl in _shuffle_levels}
    _dino_curve = {lvl: [] for lvl in _shuffle_levels}

    for _ci, (_cls, _fname) in enumerate(_demo_pool):
        _path = hf_hub_download(
            repo_id="nateraw/kinetics-mini", repo_type="dataset", filename=f"val/{_cls}/{_fname}"
        )
        _frames_64 = _decode_64(_path)
        _f0_vjepa = _vjepa_feat(_frames_64)
        _f0_dino = _dino_feat(_frames_64)
        for _lvl in _shuffle_levels:
            _shuffled = _partial_shuffle(_frames_64, _lvl, seed=100 + _ci * 10 + _lvl)
            _fv = _vjepa_feat(_shuffled)
            _fd = _dino_feat(_shuffled)
            _cos_v = float(
                np.dot(_f0_vjepa, _fv) / (np.linalg.norm(_f0_vjepa) * np.linalg.norm(_fv) + 1e-12)
            )
            _cos_d = float(
                np.dot(_f0_dino, _fd) / (np.linalg.norm(_f0_dino) * np.linalg.norm(_fd) + 1e-12)
            )
            _vjepa_curve[_lvl].append(1.0 - _cos_v)
            _dino_curve[_lvl].append(1.0 - _cos_d)

    lab_elapsed_s = _time_mod.time() - _t0
    lab_device = _device
    lab_shuffle_levels = _shuffle_levels
    lab_vjepa_mean = [float(np.mean(_vjepa_curve[l])) for l in _shuffle_levels]
    lab_dino_mean = [float(np.mean(_dino_curve[l])) for l in _shuffle_levels]
    lab_n_clips = len(_demo_pool)

    mo.md(f"Done: {lab_n_clips} clip(s) x {len(_shuffle_levels)} shuffle levels in {lab_elapsed_s:.1f}s on `{_device}`.")
    return lab_device, lab_dino_mean, lab_elapsed_s, lab_n_clips, lab_shuffle_levels, lab_vjepa_mean


@app.cell(hide_code=True)
def _(lab_dino_mean, lab_n_clips, lab_shuffle_levels, lab_vjepa_mean, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(lab_shuffle_levels, lab_vjepa_mean, "o-", color="#2166ac", linewidth=2, label="V-JEPA 2 (video-pretrained)")
    _ax.plot(lab_shuffle_levels, lab_dino_mean, "o-", color="#b2182b", linewidth=2, label="DINOv2 (image-pretrained, mean-pooled)")
    _ax.set_xlabel("shuffle strength (% of frames randomly permuted)")
    _ax.set_ylabel("mean cosine distance from unshuffled features")
    _ax.set_title(f"Order-sensitivity grows with shuffle strength ({lab_n_clips} clip(s))")
    _ax.legend()
    _ax.grid(alpha=0.25)
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ### Running this notebook yourself

        This notebook is self-contained — the frozen numbers and figures above
        are embedded directly, no re-run required. To explore or edit it locally:

        ```sh
        marimo edit notebooks/vjepa2_temporal_order_reproduction.py   # interactive editing
        marimo run notebooks/vjepa2_temporal_order_reproduction.py    # read-only app view
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
