#!/usr/bin/env python
"""Crop and downscale a LeRobotDataset's videos to the resolution the model expects.

Writes a NEW dataset whose videos have been cropped to the workspace and resized to
a square, leaving `data/` and every non-image metadata field untouched. Frame counts
and timestamps are preserved exactly, so all episode indexing stays valid.

Why this is necessary rather than a training-time transform:

  * `SpatialLearnedEmbeddings(height=4, width=4, ...)` in the reward classifier is
    hardcoded, and ResNet10 downsamples by 32 -- so the encoder only accepts
    128x128 input. Full-resolution frames produce a 20x15 feature map and fail.
  * Nothing in the classifier training path crops or resizes:
    `make_classifier_processor` builds only Normalizer -> Device,
    `RewardClassifierConfig` has no image-size fields, `ImageTransformsConfig` is
    random augmentation, and `reencode_dataset` only changes encoder parameters.
  * At RL runtime `ImageCropResizeProcessorStep` runs *before* the classifier in the
    env pipeline, so the classifier only ever sees cropped, resized frames. Training
    on anything else is a silent train/serve mismatch -- nothing errors, the reward
    is just wrong, and SAC faithfully learns from it.

Baking the crop into the dataset fixes the geometry once, at the source. Classifier
training, the offline RL buffer, and the replay-buffer memory budget all inherit it.

The crop boxes were chosen by inspecting real frames at both the approach and the
lifted "hold" pose. The side camera needs the full height of the workspace column
(the arm rises ~106 deg after grasping) but only the left third horizontally, where
the tube always sits. The wrist camera is already almost entirely workspace, so it
is resized without cropping.

Encoder parameters are copied from the source, which matters most for `video.g`:
the datasets use a 2-frame GOP so that random frame access stays fast. Re-encoding
with a default GOP would make every random read decode up to 161 frames.

Example:
    python scripts/learning/crop_dataset.py \
        --src Atomictan/grab_tube_reward_v1 \
        --dst Atomictan/grab_tube_reward_v1_128 \
        --crop observation.images.side=0,0,440,300 \
        --size 128
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}

CODEC_TO_ENCODER = {"av1": "libsvtav1", "h264": "libx264", "hevc": "libx265"}


def parse_crop(spec: str) -> tuple[str, tuple[int, int, int, int]]:
    """Parse 'observation.images.side=top,left,height,width'."""
    key, _, box = spec.partition("=")
    values = tuple(int(v) for v in box.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"crop must be top,left,height,width -- got {box!r}")
    return key, values


def build_filter(crop: tuple[int, int, int, int] | None, size: int) -> str:
    """torchvision-style (top, left, height, width) -> ffmpeg's crop=w:h:x:y."""
    if crop is None:
        return f"scale={size}:{size}"
    top, left, height, width = crop
    return f"crop={width}:{height}:{left}:{top},scale={size}:{size}"


def encoder_flags(video_info: dict) -> list[str]:
    codec = video_info.get("video.codec", "av1")
    flags = ["-c:v", CODEC_TO_ENCODER.get(codec, "libsvtav1")]
    for info_key, flag in (("video.crf", "-crf"), ("video.preset", "-preset"), ("video.g", "-g")):
        value = video_info.get(info_key)
        if value is not None:
            flags += [flag, str(value)]
    flags += ["-pix_fmt", video_info.get("video.pix_fmt", "yuv420p")]
    return flags


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def image_stats(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Per-channel stats in LeRobot's (3, 1, 1) layout. `frames` is (N, C, H, W) in [0, 1]."""
    flat = frames.reshape(frames.shape[0], frames.shape[1], -1).transpose(1, 0, 2).reshape(frames.shape[1], -1)
    stats = {
        "min": flat.min(axis=1).reshape(3, 1, 1),
        "max": flat.max(axis=1).reshape(3, 1, 1),
        "mean": flat.mean(axis=1).reshape(3, 1, 1),
        "std": flat.std(axis=1).reshape(3, 1, 1),
        "count": np.array([flat.shape[1] * flat.shape[0] // 3]),
    }
    for key, q in QUANTILES.items():
        stats[key] = np.quantile(flat, q, axis=1).reshape(3, 1, 1)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="source repo_id")
    parser.add_argument("--dst", required=True, help="destination repo_id")
    parser.add_argument(
        "--crop",
        action="append",
        default=[],
        type=parse_crop,
        metavar="KEY=TOP,LEFT,HEIGHT,WIDTH",
        help="crop box for one camera key (repeatable); keys not listed are resized only",
    )
    parser.add_argument("--size", type=int, default=128, help="output square edge (default 128)")
    parser.add_argument(
        "--stats-frames",
        type=int,
        default=25,
        help="frames sampled per episode to recompute image stats (0 to skip)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="recompute image stats on an already-written destination; skips copy and re-encode",
    )
    args = parser.parse_args()

    crops = dict(args.crop)
    src_root = LEROBOT_HOME / args.src
    dst_root = LEROBOT_HOME / args.dst
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    if args.stats_only:
        if not dst_root.is_dir():
            raise FileNotFoundError(f"{dst_root} does not exist; run without --stats-only first")
    elif dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists (pass --overwrite)")
        shutil.rmtree(dst_root)

    # On a --stats-only rerun the destination already declares the new geometry.
    info = json.loads(((dst_root if args.stats_only else src_root) / "meta" / "info.json").read_text())
    camera_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    unknown = set(crops) - set(camera_keys)
    if unknown:
        raise ValueError(f"--crop names unknown camera keys: {sorted(unknown)}; have {camera_keys}")

    print(f"{args.src} -> {args.dst}   size {args.size}x{args.size}")
    for key in camera_keys:
        box = crops.get(key)
        print(f"    {key:32} {'crop ' + str(box) if box else 'no crop'}  filter: {build_filter(box, args.size)}")

    if args.stats_only:
        recompute_image_stats(args.dst, dst_root, camera_keys, args.stats_frames)
        verify(args.dst, dst_root, args.size)
        return

    # data/ and meta/ carry no pixel dimensions -- copy verbatim, then patch info.json.
    shutil.copytree(src_root / "data", dst_root / "data")
    shutil.copytree(src_root / "meta", dst_root / "meta")

    started = time.perf_counter()
    total_in = total_out = 0
    for key in camera_keys:
        flags = encoder_flags(info["features"][key].get("info", {}))
        filt = build_filter(crops.get(key), args.size)
        for src_video in sorted((src_root / "videos" / key).rglob("*.mp4")):
            dst_video = dst_root / "videos" / key / src_video.relative_to(src_root / "videos" / key)
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_video), "-vf", filt,
                 *flags, str(dst_video)],
                check=True, capture_output=True,
            )
            n_in, n_out = count_frames(src_video), count_frames(dst_video)
            if n_in != n_out:
                raise RuntimeError(f"frame count changed for {src_video.name}: {n_in} -> {n_out}")
            total_in += src_video.stat().st_size
            total_out += dst_video.stat().st_size
            print(f"    {key.split('.')[-1]:6} {src_video.name}  {n_in:>6} frames  "
                  f"{src_video.stat().st_size / 1e6:>7.1f} MB -> {dst_video.stat().st_size / 1e6:>6.1f} MB")

    # info.json: declare the new geometry.
    for key in camera_keys:
        info["features"][key]["shape"] = [args.size, args.size, 3]
        if "info" in info["features"][key]:
            info["features"][key]["info"]["video.height"] = args.size
            info["features"][key]["info"]["video.width"] = args.size
    (dst_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"\nvideo {total_in / 1e9:.2f} GB -> {total_out / 1e9:.2f} GB "
          f"({total_in / max(total_out, 1):.0f}x smaller) in {time.perf_counter() - started:.0f}s")

    if args.stats_frames > 0:
        recompute_image_stats(args.dst, dst_root, camera_keys, args.stats_frames)

    verify(args.dst, dst_root, args.size)


def recompute_image_stats(repo_id: str, root: Path, camera_keys: list[str], per_episode: int) -> None:
    """Cropping changes the pixel distribution, so the stored image stats are now wrong.

    Training does not actually consume these -- `use_imagenet_stats` defaults to True and
    overwrites camera stats with ImageNet values -- but leaving stats that describe the
    uncropped frames is a trap for anything that later aggregates or inspects this dataset.
    Sampled rather than exhaustive: a few thousand frames pin per-channel moments tightly.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"\nrecomputing image stats from {per_episode} frames/episode ...")
    dataset = LeRobotDataset(repo_id, root=root)
    episode_index = np.asarray(dataset.hf_dataset["episode_index"])
    rng = np.random.default_rng(0)

    per_ep_stats: dict[str, dict[int, dict[str, np.ndarray]]] = {k: {} for k in camera_keys}
    pooled: dict[str, list[np.ndarray]] = {k: [] for k in camera_keys}

    for episode in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == episode)
        picks = rng.choice(rows, size=min(per_episode, len(rows)), replace=False)
        samples = [dataset[int(i)] for i in picks]
        for key in camera_keys:
            frames = np.stack([s[key].numpy() for s in samples])
            per_ep_stats[key][int(episode)] = image_stats(frames)
            pooled[key].append(frames)

    stats = json.loads((root / "meta" / "stats.json").read_text())
    for key in camera_keys:
        aggregate = image_stats(np.concatenate(pooled[key]))
        stats[key] = {k: v.tolist() for k, v in aggregate.items()}
        print(f"    {key:32} mean {np.array(aggregate['mean']).ravel().round(3)}  "
              f"std {np.array(aggregate['std']).ravel().round(3)}")
    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        table = pd.read_parquet(path)
        for key in camera_keys:
            for stat in STAT_KEYS:
                # pyarrow stores these as nested lists (list<list<list<double>>>) and
                # rejects a 3-D ndarray outright, so hand it Python lists.
                table[f"stats/{key}/{stat}"] = [
                    per_ep_stats[key][int(e)][stat].tolist() for e in table["episode_index"]
                ]
        table.to_parquet(path, index=False)


def verify(repo_id: str, root: Path, size: int) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import REWARD

    dataset = LeRobotDataset(repo_id, root=root)
    rewards = np.asarray(dataset.hf_dataset[REWARD], dtype=np.float32)
    positive = np.flatnonzero(rewards > 0)
    print(f"\nverify: {len(dataset):,} frames, {len(positive):,} positive")
    for idx in (0, int(positive[len(positive) // 2]), len(dataset) - 1):
        sample = dataset[idx]
        shapes = {k.split(".")[-1]: tuple(v.shape) for k, v in sample.items() if k.startswith("observation.images")}
        assert all(s[-2:] == (size, size) for s in shapes.values()), f"bad shape at {idx}: {shapes}"
        print(f"    idx {idx:>7}  ep {int(sample['episode_index']):>3}  "
              f"reward {float(sample[REWARD]):.0f}  {shapes}")
    print("all sampled frames decode at the target resolution")


if __name__ == "__main__":
    main()
