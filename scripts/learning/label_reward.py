#!/usr/bin/env python
"""Add a binary `next.reward` column to SO-101 grasp datasets.

Writes a NEW dataset alongside the source; the source is never modified. The
`videos/` directory is symlinked rather than copied, so labelling a 2.5 GB
dataset costs a few MB of parquet and JSON.

The rule reads motor data only -- never pixels:

    positive  <=>  action[gripper]            <  min + 0.20 * range  (per episode)
              AND  observation.state[gripper] >= 6                   (absolute)
              AND  that holds for a run of >= 60 frames (2.0 s @ 30 fps),
                   bridging dropouts of <= 5 frames
              AND  |shoulder_pan - shoulder_pan_at_grasp| < 20 deg

The two bounds deliberately read different signals. The *commanded* position
says the operator wants the jaws shut; the *achieved* position says they did not
actually shut. Therefore something is between them. Reading both from the same
signal fails: on commanded alone the floor rejects real grasps (the operator
keeps squeezing, so the command goes to ~0-3 while the jaws stop at 10-13); on
achieved alone the closed threshold lands at ~8.2, below the band a held tube
occupies.

The pan term scopes a positive to the pick site, so carrying the tube to the box
is not labelled. It has no effect on grasp *detection* -- it only decides where
the pick sub-task ends.

Datasets passed with --negative are forced to all-zero regardless of the rule.
Those are deliberately recorded failures where ground truth is known by
construction. The rule is still evaluated on them and disagreements are
reported: that is how we find holes in it. A known, expected disagreement is
closing on the box rim, which stops the jaws at ~9 -- inside the tube's band.
Proprioception cannot separate those two cases, which is the whole reason the
labelled data goes on to train a vision classifier.

Example:
    python scripts/learning/label_reward.py \
        --dataset Atomictan/grab_tube_merged_v3 \
        --dataset Atomictan/rollout_eval_v4_20260722_105932 \
        --dataset Atomictan/rollout_eval_v4_20260722_121904 \
        --dataset Atomictan/rollout_eval_v5_20260722_192826 \
        --negative Atomictan/grab_tube_negatives_v1_20260729_225530
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.utils.constants import ACTION, OBS_STATE, REWARD

# Motor layout for the SO-101 follower: see meta/info.json "names".
GRIPPER_IDX = 5
PAN_IDX = 0

# Rule parameters. Every one of these was fixed empirically -- see the module
# docstring and the grasp-improvement writeup for the supporting measurements.
CLOSED_FRAC = 0.20  # threshold plateau is [0.20, 0.25]; below 0.15 loses demos
APERTURE_FLOOR = 6.0  # empty-air closures reach ~1-3; a held tube sits at 10-13
MIN_RUN_FRAMES = 60  # 2.0 s at 30 fps; real grasps last a median of 184 frames
RUN_GAP_TOLERANCE = 5  # bridge dropouts so one noisy frame cannot split a hold
PAN_TOLERANCE_DEG = 20.0  # ~2.8 s of hold; the arm swings ~106 deg to the box

STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}

LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


def label_episode(action: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Return a per-frame boolean mask of "the tube is grasped, at the pick site"."""
    commanded = action[:, GRIPPER_IDX]
    achieved = state[:, GRIPPER_IDX]
    pan = action[:, PAN_IDX]

    labels = np.zeros(len(commanded), dtype=bool)

    low, high = commanded.min(), commanded.max()
    if high - low < 1e-6:
        # Gripper never moved; nothing to normalise against.
        return labels

    threshold = low + CLOSED_FRAC * (high - low)
    in_band = (commanded < threshold) & (achieved >= APERTURE_FLOOR)

    indices = np.flatnonzero(in_band)
    if indices.size == 0:
        return labels

    breaks = np.where(np.diff(indices) > RUN_GAP_TOLERANCE)[0] + 1
    for run in np.split(indices, breaks):
        if run.size < MIN_RUN_FRAMES:
            continue
        pan_at_grasp = pan[run[0]]
        within_scope = run[np.abs(pan[run] - pan_at_grasp) < PAN_TOLERANCE_DEG]
        labels[within_scope] = True

    return labels


def feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """Build the stats block LeRobot expects for a scalar feature."""
    values = np.asarray(values, dtype=np.float64)
    stats = {
        "min": np.array([values.min()]),
        "max": np.array([values.max()]),
        "mean": np.array([values.mean()]),
        "std": np.array([values.std()]),
        "count": np.array([values.size]),
    }
    for key, q in QUANTILES.items():
        stats[key] = np.array([np.quantile(values, q)])
    return stats


def _stack(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(frame[column].to_numpy())


def label_dataset(
    repo_id: str,
    suffix: str,
    force_negative: bool,
    dry_run: bool,
    overwrite: bool,
) -> dict:
    src_root = LEROBOT_HOME / repo_id
    if not src_root.is_dir():
        raise FileNotFoundError(f"dataset not found: {src_root}")

    namespace, name = repo_id.split("/", 1)
    out_repo_id = f"{namespace}/{name}{suffix}"
    out_root = LEROBOT_HOME / out_repo_id

    info = json.loads((src_root / "meta" / "info.json").read_text())
    if REWARD in info["features"]:
        raise ValueError(f"{repo_id} already has a {REWARD} column")

    data_files = sorted((src_root / "data").rglob("*.parquet"))
    frames = []
    for path in data_files:
        chunk = pd.read_parquet(path)
        chunk["__source"] = str(path.relative_to(src_root))
        frames.append(chunk)
    data = pd.concat(frames, ignore_index=True)

    # Label per episode, then report both the rule's verdict and what we store.
    rewards = np.zeros(len(data), dtype=np.float32)
    episodes = []
    for episode_index, group in data.groupby("episode_index", sort=True):
        order = group.sort_values("frame_index")
        rule = label_episode(_stack(order, ACTION), _stack(order, OBS_STATE))
        stored = np.zeros_like(rule) if force_negative else rule
        rewards[order.index.to_numpy()] = stored.astype(np.float32)
        episodes.append(
            {
                "episode": int(episode_index),
                "frames": len(order),
                "rule_positive": int(rule.sum()),
                "stored_positive": int(stored.sum()),
            }
        )

    data[REWARD] = rewards

    summary = {
        "repo_id": repo_id,
        "out_repo_id": out_repo_id,
        "episodes": episodes,
        "total_frames": len(data),
        "total_positive": int((rewards > 0).sum()),
        "episodes_with_positive": sum(1 for e in episodes if e["stored_positive"] > 0),
        "rule_disagreements": [e["episode"] for e in episodes if force_negative and e["rule_positive"] > 0],
        "forced_negative": force_negative,
    }

    if dry_run:
        return summary

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists (pass --overwrite to replace)")
        shutil.rmtree(out_root)

    # data/ -- rewrite each source parquet with the extra column, same layout.
    for relative, group in data.groupby("__source", sort=True):
        destination = out_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        group.drop(columns="__source").to_parquet(destination, index=False)

    # videos/ -- symlink, never copy. Absolute target so it resolves from anywhere.
    src_videos = src_root / "videos"
    if src_videos.is_dir():
        (out_root / "videos").symlink_to(src_videos.resolve(), target_is_directory=True)

    meta_out = out_root / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_root / "meta" / "tasks.parquet", meta_out / "tasks.parquet")

    # meta/info.json -- declare the new feature.
    info["features"][REWARD] = {"dtype": "float32", "shape": [1], "names": None}
    (meta_out / "info.json").write_text(json.dumps(info, indent=4))

    # meta/stats.json -- dataset-wide stats for the new feature.
    stats = json.loads((src_root / "meta" / "stats.json").read_text())
    stats[REWARD] = {key: value.tolist() for key, value in feature_stats(rewards).items()}
    (meta_out / "stats.json").write_text(json.dumps(stats, indent=4))

    # meta/episodes/*.parquet -- per-episode stats, as flat "stats/<feature>/<key>" columns.
    per_episode = {
        int(episode_index): feature_stats(group[REWARD].to_numpy())
        for episode_index, group in data.groupby("episode_index", sort=True)
    }
    for path in sorted((src_root / "meta" / "episodes").rglob("*.parquet")):
        table = pd.read_parquet(path)
        for key in STAT_KEYS:
            table[f"stats/{REWARD}/{key}"] = [
                per_episode[int(episode_index)][key] for episode_index in table["episode_index"]
            ]
        destination = meta_out / "episodes" / path.relative_to(src_root / "meta" / "episodes")
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(destination, index=False)

    return summary


def verify(out_repo_id: str) -> str:
    """Load the written dataset the way training will, and confirm the column survives."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(out_repo_id, root=LEROBOT_HOME / out_repo_id)
    sample = dataset[0]
    if REWARD not in sample:
        raise AssertionError(f"{REWARD} missing after reload")
    return f"loaded {len(dataset)} frames, {REWARD}={float(sample[REWARD]):.0f}, keys ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="REPO_ID",
        help="dataset to label with the rule (repeatable)",
    )
    parser.add_argument(
        "--negative",
        action="append",
        default=[],
        metavar="REPO_ID",
        help="dataset to force to all-zero reward (repeatable)",
    )
    parser.add_argument("--suffix", default="_labeled", help="appended to each output repo_id")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output dataset")
    parser.add_argument("--per-episode", action="store_true", help="print every episode, not just a summary")
    args = parser.parse_args()

    if not args.dataset and not args.negative:
        parser.error("pass at least one --dataset or --negative")

    jobs = [(repo_id, False) for repo_id in args.dataset]
    jobs += [(repo_id, True) for repo_id in args.negative]

    summaries = []
    for repo_id, force_negative in jobs:
        summary = label_dataset(
            repo_id=repo_id,
            suffix=args.suffix,
            force_negative=force_negative,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        summaries.append(summary)

        mode = "FORCED NEGATIVE" if force_negative else "rule"
        print(f"\n{summary['repo_id']}  [{mode}]  ->  {summary['out_repo_id']}")
        if args.per_episode:
            for episode in summary["episodes"]:
                flag = "" if episode["rule_positive"] == episode["stored_positive"] else "   <-- rule disagrees"
                print(
                    f"    ep {episode['episode']:>3}  {episode['frames']:>6} frames"
                    f"  rule {episode['rule_positive']:>5}  stored {episode['stored_positive']:>5}{flag}"
                )
        share = 100 * summary["total_positive"] / max(summary["total_frames"], 1)
        print(
            f"    {summary['total_frames']:,} frames"
            f"   {summary['total_positive']:,} positive ({share:.1f}%)"
            f"   {summary['episodes_with_positive']}/{len(summary['episodes'])} episodes with a positive"
        )
        if summary["rule_disagreements"]:
            print(
                f"    rule flagged {len(summary['rule_disagreements'])} episode(s) positive "
                f"that we store as negative: {summary['rule_disagreements']}"
            )
            print("    ^ expected for thick non-tube objects (e.g. the box rim). These are the hard negatives.")

    total_frames = sum(s["total_frames"] for s in summaries)
    total_positive = sum(s["total_positive"] for s in summaries)
    print(
        f"\nTOTAL  {total_frames:,} frames   {total_positive:,} positive "
        f"({100 * total_positive / max(total_frames, 1):.1f}%)   "
        f"{total_frames - total_positive:,} negative"
    )

    if args.dry_run:
        print("\ndry run -- nothing written")
        return

    print("\nverifying each output loads through LeRobotDataset ...")
    for summary in summaries:
        print(f"    {summary['out_repo_id']}: {verify(summary['out_repo_id'])}")


if __name__ == "__main__":
    main()
