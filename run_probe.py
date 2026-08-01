"""
Reproduction of V-JEPA 2's motion-vs-appearance claim (arXiv:2506.09985, Sec 5.1),
downscaled to a real-order-vs-shuffled-order probe on frozen features.

Two frozen encoders see the same 64 sampled frames of each clip, once in the
correct order and once with those exact same frames permuted:

  - facebook/dinov2-small : encodes each frame INDEPENDENTLY, then mean-pools
    across frames. Mean-pooling a set is provably invariant to the order the
    set was visited in, so this encoder's real-vs-shuffled features must be
    numerically identical (up to floating point). A linear probe trained to
    tell them apart cannot do better than chance -- this is checked directly,
    not assumed.

  - facebook/vjepa2-vitl-fpc64-256 : encodes all 64 frames JOINTLY through
    spatio-temporal self-attention with positional structure, so permuting
    frame order changes which tokens attend to which positions. If the paper's
    claim holds at this tiny scale, a linear probe on its pooled features
    should beat chance at detecting the shuffle.

Prints one JSON summary block to stdout at the end -- the only evidence
channel this run has (local-mode logs).
"""

import json
import time

import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel

# Manual replacement for AutoVideoProcessor (which hard-requires torchvision as a
# backend) -- values copied verbatim from vjepa2-vitl-fpc64-256's own
# video_preprocessor_config.json: resize shortest edge to 292, center-crop to
# 256x256, rescale to [0,1], normalize with ImageNet mean/std. Implemented with
# PIL + numpy only so the run command's dependency list doesn't need touching.
VJEPA_RESIZE_SHORT_EDGE = 292
VJEPA_CROP_SIZE = 256
VJEPA_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
VJEPA_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEED = 42
NUM_FRAMES = 64          # fixed by the vjepa2-vitl-fpc64-256 checkpoint
BASELINE_BATCH = 16      # chunk size only (memory), NOT a subsample -- all 64 frames are pooled
DATASET_REPO = "nateraw/kinetics-mini"
VJEPA_REPO = "facebook/vjepa2-vitl-fpc64-256"
BASELINE_REPO = "facebook/dinov2-small"

CLASSES = ["archery", "bowling", "flying_kite", "high_jump", "marching"]

# 2 clips/class from train/ (probe-train) + 2 clips/class from val/ (probe-test, disjoint videos)
TRAIN_CLIPS = {
    "archery": ["-1q7jA3DXQM_000005_000015.mp4", "-5NN5hdIwTc_000036_000046.mp4"],
    "bowling": ["-5ExwuF5IUI_000030_000040.mp4", "-7sTNNI1Bcg_000075_000085.mp4"],
    "flying_kite": ["-0yuyrbruYM_000079_000089.mp4", "-4dHxTY4iwo_000043_000053.mp4"],
    "high_jump": ["-B228_dxIVc_000001_000011.mp4", "-D_eNsRYLhs_000003_000013.mp4"],
    "marching": ["-0kVxoFuMGQ_000622_000632.mp4", "-2KG4xiBL1s_000005_000015.mp4"],
}
TEST_CLIPS = {
    "archery": ["-Qz25rXdMjE_000014_000024.mp4", "-UJgyiWe500_000029_000039.mp4"],
    "bowling": ["--dVV4_CSvw_000033_000043.mp4", "-WH-lxmGJVY_000005_000015.mp4"],
    "flying_kite": ["07xOT83TIG4_000040_000050.mp4", "0QH8uFjXiW4_000003_000013.mp4"],
    "high_jump": ["01fAWEHzudA_000002_000012.mp4", "0oL36GHlSXw_000022_000032.mp4"],
    "marching": ["-0IErS_cisg_000017_000027.mp4", "1m-Kdky1y84_000022_000032.mp4"],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def decode_all_frames(path):
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames, axis=0)  # (T, H, W, C) uint8


def sample_64(frames):
    t = frames.shape[0]
    idx = np.linspace(0, t - 1, NUM_FRAMES).round().astype(int)
    idx = np.clip(idx, 0, t - 1)
    return frames[idx]  # (64, H, W, C)


def shuffled_index(clip_seed):
    rng = np.random.RandomState(clip_seed)
    return rng.permutation(NUM_FRAMES)


def preprocess_frame_for_vjepa(frame_hwc_uint8):
    img = Image.fromarray(frame_hwc_uint8)
    w, h = img.size
    if w <= h:
        new_w = VJEPA_RESIZE_SHORT_EDGE
        new_h = int(round(h * VJEPA_RESIZE_SHORT_EDGE / w))
    else:
        new_h = VJEPA_RESIZE_SHORT_EDGE
        new_w = int(round(w * VJEPA_RESIZE_SHORT_EDGE / h))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - VJEPA_CROP_SIZE) // 2
    top = (new_h - VJEPA_CROP_SIZE) // 2
    img = img.crop((left, top, left + VJEPA_CROP_SIZE, top + VJEPA_CROP_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - VJEPA_IMAGE_MEAN) / VJEPA_IMAGE_STD
    return arr.transpose(2, 0, 1)  # C, H, W


def vjepa_pooled_features(model, device, frames_64):
    # frames_64: (64, H, W, C) uint8 -> manually preprocessed (1, T, C, H, W) float
    chw = [preprocess_frame_for_vjepa(f) for f in frames_64]
    video = torch.from_numpy(np.stack(chw, axis=0)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.get_vision_features(pixel_values_videos=video)
    return feats.mean(dim=1).squeeze(0).cpu().numpy()


def baseline_pooled_features(model, processor, device, frames_64):
    # Pools over ALL 64 frames (the exact same set for "real" and "shuffled" --
    # just a different order), processed independently in chunks then averaged.
    # Mean-of-means over equal-size chunks == mean over the full set, so chunking
    # for memory doesn't change the invariance argument.
    running_sum = 0.0
    running_count = 0
    for start in range(0, NUM_FRAMES, BASELINE_BATCH):
        chunk = frames_64[start:start + BASELINE_BATCH]
        imgs = [Image.fromarray(f) for f in chunk]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs).last_hidden_state  # (chunk_len, tokens, hidden)
        running_sum = running_sum + out.sum(dim=(0, 1)).cpu().numpy()
        running_count += out.shape[0] * out.shape[1]
    return running_sum / running_count


def build_split(clip_map, split_name, vjepa, dino, dino_proc, device, clip_counter):
    X_vjepa, y_vjepa = [], []
    X_base, y_base = [], []
    max_base_diff = 0.0
    for cls in CLASSES:
        for fname in clip_map[cls]:
            clip_counter["n"] += 1
            seed_here = SEED + clip_counter["n"]
            t0 = time.time()
            local_path = hf_hub_download(
                repo_id=DATASET_REPO, repo_type="dataset",
                filename=f"{split_name}/{cls}/{fname}",
            )
            frames = decode_all_frames(local_path)
            frames_64 = sample_64(frames)
            perm = shuffled_index(seed_here)
            frames_shuffled = frames_64[perm]

            f_real = vjepa_pooled_features(vjepa, device, frames_64)
            f_shuf = vjepa_pooled_features(vjepa, device, frames_shuffled)
            X_vjepa += [f_real, f_shuf]
            y_vjepa += [1, 0]

            b_real = baseline_pooled_features(dino, dino_proc, device, frames_64)
            b_shuf = baseline_pooled_features(dino, dino_proc, device, frames_shuffled)
            X_base += [b_real, b_shuf]
            y_base += [1, 0]
            max_base_diff = max(max_base_diff, float(np.abs(b_real - b_shuf).max()))

            dt = time.time() - t0
            log(f"{split_name}/{cls}/{fname}: {frames.shape[0]} frames decoded, "
                f"vjepa+baseline pass done in {dt:.1f}s")
    return (np.array(X_vjepa), np.array(y_vjepa),
            np.array(X_base), np.array(y_base), max_base_diff)


def main():
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    log(f"loading {VJEPA_REPO} ...")
    vjepa = AutoModel.from_pretrained(VJEPA_REPO).to(device).eval()

    log(f"loading {BASELINE_REPO} ...")
    dino_proc = AutoImageProcessor.from_pretrained(BASELINE_REPO, use_fast=False)
    dino = AutoModel.from_pretrained(BASELINE_REPO).to(device).eval()

    clip_counter = {"n": 0}
    log("building probe-train split (train/, 2 clips x 5 classes) ...")
    Xv_tr, yv_tr, Xb_tr, yb_tr, diff_tr = build_split(
        TRAIN_CLIPS, "train", vjepa, dino, dino_proc, device, clip_counter)
    log("building probe-test split (val/, 2 clips x 5 classes, disjoint videos) ...")
    Xv_te, yv_te, Xb_te, yb_te, diff_te = build_split(
        TEST_CLIPS, "val", vjepa, dino, dino_proc, device, clip_counter)

    max_base_diff = max(diff_tr, diff_te)

    probe_vjepa = LogisticRegression(max_iter=2000).fit(Xv_tr, yv_tr)
    vjepa_train_acc = float(probe_vjepa.score(Xv_tr, yv_tr))
    vjepa_test_acc = float(probe_vjepa.score(Xv_te, yv_te))

    probe_base = LogisticRegression(max_iter=2000).fit(Xb_tr, yb_tr)
    base_train_acc = float(probe_base.score(Xb_tr, yb_tr))
    base_test_acc = float(probe_base.score(Xb_te, yb_te))

    # cosine distance between real/shuffled vjepa features, mean over all clips (train+test)
    Xv_all = np.concatenate([Xv_tr, Xv_te], axis=0)
    yv_all = np.concatenate([yv_tr, yv_te], axis=0)
    reals = Xv_all[yv_all == 1]
    shufs = Xv_all[yv_all == 0]
    cos = np.sum(reals * shufs, axis=1) / (
        np.linalg.norm(reals, axis=1) * np.linalg.norm(shufs, axis=1) + 1e-12)
    vjepa_mean_cosine_dist = float(np.mean(1.0 - cos))

    n_clips = clip_counter["n"]
    total_wall_s = time.time() - t_start

    summary = {
        "claim": "video-pretrained frozen features (V-JEPA 2) are order-sensitive; "
                 "independently-encoded-then-mean-pooled image features (DINOv2) are not",
        "paper": "arXiv:2506.09985 Sec 5.1 (SSv2 motion-probe headline: ViT-g 77.3% vs "
                 "image/other encoders 55.4-69.7%) -- downscaled proxy below, not a claim of matching those numbers",
        "vjepa_model": VJEPA_REPO,
        "baseline_model": BASELINE_REPO,
        "dataset": DATASET_REPO,
        "n_clips_train": sum(len(v) for v in TRAIN_CLIPS.values()),
        "n_clips_test": sum(len(v) for v in TEST_CLIPS.values()),
        "n_clips_total": n_clips,
        "n_probe_examples_train": int(len(yv_tr)),
        "n_probe_examples_test": int(len(yv_te)),
        "frames_per_clip": NUM_FRAMES,
        "baseline_frames_per_clip": NUM_FRAMES,
        "seed": SEED,
        "device": device,
        "vjepa_probe_train_acc": vjepa_train_acc,
        "vjepa_probe_test_acc": vjepa_test_acc,
        "baseline_probe_train_acc": base_train_acc,
        "baseline_probe_test_acc": base_test_acc,
        "baseline_max_abs_feature_diff_real_vs_shuffled": max_base_diff,
        "vjepa_mean_cosine_distance_real_vs_shuffled": vjepa_mean_cosine_dist,
        "total_wall_time_s": round(total_wall_s, 1),
    }

    print("=" * 70, flush=True)
    print("RESULT_SUMMARY_JSON", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("=" * 70, flush=True)

    log("human-readable summary:")
    log(f"  baseline (DINOv2, order-invariant by construction): "
        f"train acc={base_train_acc:.3f} test acc={base_test_acc:.3f} "
        f"(expect ~0.5; max|real-shuffled| feature diff={max_base_diff:.2e}, expect ~0)")
    log(f"  V-JEPA 2 (joint spatio-temporal attention): "
        f"train acc={vjepa_train_acc:.3f} test acc={vjepa_test_acc:.3f} "
        f"(expect > baseline if motion/order info is captured)")
    log(f"  mean cosine distance real-vs-shuffled, V-JEPA 2 features: {vjepa_mean_cosine_dist:.4f}")
    log(f"  total wall time: {total_wall_s:.1f}s over {n_clips} clips "
        f"({2 * n_clips} vjepa forward passes)")


if __name__ == "__main__":
    main()
