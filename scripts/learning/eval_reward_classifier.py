#!/usr/bin/env python
"""Score a trained reward classifier the way HIL-SERL will actually use it.

`lerobot-train` logs only `eval_loss`, which is close to useless here: the labels are
~8.8% positive, so a model that predicts 0 for every frame scores 91.2% accuracy and a
respectable loss while being completely worthless as a reward signal. This reports the
numbers that decide whether the classifier is safe to run RL on.

Three views, in increasing order of how much they matter:

  per-frame precision/recall
      Swept across thresholds, because `success_threshold` is a free runtime knob
      (`envs/configs.py`) -- the precision/recall trade can be re-picked after training
      without retraining anything.

  episode-level TP/TN/FP/FN against human verdicts
      An episode counts as "detected" if ANY frame fires, matching HIL-SERL's
      terminate-on-first-positive behaviour. This is the number directly comparable to
      the hand-written rule's 62/6/2/0, and it is the one to judge the model on.

  false alarms on deliberate-negative episodes
      Episodes recorded specifically as failures. Any firing here is a would-be reward
      hack: SAC would learn whatever produced it.

Costs are asymmetric. A false positive ends the episode and pays reward for nothing, and
the agent will reproduce whatever triggered it. A false negative merely wastes an episode.
So precision matters more than recall -- and per-frame recall matters less than it looks,
since a real grasp spans ~84 labelled frames and only one needs to fire.

Images are normalized with ImageNet stats, matching both `NormalizerProcessorStep` during
training and the patched `RewardClassifierProcessorStep` at runtime.

Example:
    python scripts/learning/eval_reward_classifier.py \
        --checkpoint outputs/train/reward_classifier_v1/checkpoints/last/pretrained_model \
        --dataset Atomictan/grab_tube_reward_v1_128 \
        --eval-split 0.2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.constants import IMAGENET_STATS, REWARD

LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"
VERDICT_DIR = Path("outputs/eval")

# The merge order that produced grab_tube_reward_v1, and the verdict log for each source.
# "NEGATIVE" means every episode is a known failure by construction.
SOURCES: list[tuple[str, str | None]] = [
    ("Atomictan/grab_tube_merged_v3_labeled", None),
    ("Atomictan/rollout_eval_v4_20260722_105932_labeled", "v4_20260722_105932"),
    ("Atomictan/rollout_eval_v4_20260722_121904_labeled", "v4_20260722_121904"),
    ("Atomictan/rollout_eval_v5_20260722_192826_labeled", "v5_20260722_192826"),
    ("Atomictan/grab_tube_negatives_v1_20260729_225530_labeled", "NEGATIVE"),
    ("Atomictan/grab_tube_negatives_v2_20260730_105212_labeled", "NEGATIVE"),
    ("Atomictan/grab_tube_negatives_v3_20260730_133524_labeled", "NEGATIVE"),
]


def build_episode_map() -> dict[int, dict]:
    """Map each merged episode index back to its source dataset and human verdict."""
    episode_map: dict[int, dict] = {}
    offset = 0
    for repo_id, verdict_key in SOURCES:
        info = json.loads((LEROBOT_HOME / repo_id / "meta" / "info.json").read_text())
        n_episodes = info["total_episodes"]

        verdicts: dict[int, bool] = {}
        if verdict_key and verdict_key != "NEGATIVE":
            log = json.loads((VERDICT_DIR / f"{verdict_key}.json").read_text())
            # verdict logs number episodes from 1; dataset episode_index starts at 0
            verdicts = {e["episode"] - 1: e["success"] for e in log["episodes"] if e.get("success") is not None}

        for local in range(n_episodes):
            episode_map[offset + local] = {
                "source": repo_id.split("/")[-1],
                "local": local,
                "verdict": False if verdict_key == "NEGATIVE" else verdicts.get(local),
                "deliberate_negative": verdict_key == "NEGATIVE",
            }
        offset += n_episodes
    return episode_map


def held_out_episodes(dataset, eval_split: float) -> list[int]:
    """Reproduce `make_train_eval_datasets`: last ceil(n * split) episodes per task."""
    tasks = dataset.meta.episodes["tasks"]
    by_task: dict[str, list[int]] = {}
    for episode in range(dataset.meta.total_episodes):
        key = tasks[episode][0] if tasks[episode] else ""
        by_task.setdefault(key, []).append(episode)

    evaluation: list[int] = []
    for episodes in by_task.values():
        n_eval = math.ceil(len(episodes) * eval_split)
        evaluation.extend(episodes[len(episodes) - n_eval :])
    return sorted(evaluation)


@torch.no_grad()
def predict(checkpoint: Path, dataset, rows: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    from lerobot.rewards.classifier.modeling_classifier import Classifier

    model = Classifier.from_pretrained(str(checkpoint)).to(device).eval()
    image_keys = [k for k in dataset.meta.features if k.startswith("observation.images")]
    mean = torch.tensor(IMAGENET_STATS["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(IMAGENET_STATS["std"], dtype=torch.float32, device=device)

    subset = torch.utils.data.Subset(dataset, rows.tolist())
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    probabilities = []
    for batch in loader:
        images = [((batch[k].to(device, non_blocking=True) - mean) / std) for k in image_keys]
        output = model.predict(images)
        probabilities.append(output.probabilities.flatten().float().cpu().numpy())
    return np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", default="Atomictan/grab_tube_reward_v1_128")
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--include-train", action="store_true", help="also score training episodes")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset, root=LEROBOT_HOME / args.dataset)
    episode_map = build_episode_map()
    evaluation_episodes = held_out_episodes(dataset, args.eval_split)
    scored = (
        list(range(dataset.meta.total_episodes)) if args.include_train else evaluation_episodes
    )

    episode_index = np.asarray(dataset.hf_dataset["episode_index"])
    labels_all = np.asarray(dataset.hf_dataset[REWARD], dtype=np.float32)
    rows = np.flatnonzero(np.isin(episode_index, scored))

    print(f"checkpoint : {args.checkpoint}")
    print(f"dataset    : {args.dataset}")
    print(f"scoring    : {len(scored)} episodes, {len(rows):,} frames "
          f"({int((labels_all[rows] > 0).sum()):,} positive, "
          f"{100 * (labels_all[rows] > 0).mean():.1f}%)")

    probabilities = predict(args.checkpoint, dataset, rows, args.batch_size, args.device)
    labels = labels_all[rows] > 0
    episodes_of_row = episode_index[rows]

    print("\nPER-FRAME")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'F1':>7} {'pred pos':>10}")
    print("-" * 46)
    for threshold in args.thresholds:
        fired = probabilities > threshold
        tp = int((fired & labels).sum())
        fp = int((fired & ~labels).sum())
        fn = int((~fired & labels).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"{threshold:>7.2f} {precision:>10.3f} {recall:>8.3f} {f1:>7.3f} {int(fired.sum()):>10,}")

    print("\nPER-EPISODE  (fires if ANY frame exceeds threshold, as HIL-SERL terminates on first positive)")
    print(f"{'thresh':>7} | {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3} | {'verdict eps':>11} | "
          f"{'false alarms on deliberate negatives':>36}")
    print("-" * 96)
    for threshold in args.thresholds:
        tp = tn = fp = fn = 0
        n_verdict = 0
        false_alarms = []
        for episode in scored:
            mask = episodes_of_row == episode
            if not mask.any():
                continue
            fired = bool((probabilities[mask] > threshold).any())
            meta = episode_map.get(episode, {})
            if meta.get("deliberate_negative"):
                if fired:
                    false_alarms.append(f"{meta['source'][:12]}#{meta['local']}")
                continue
            verdict = meta.get("verdict")
            if verdict is None:
                continue
            n_verdict += 1
            if fired and verdict:
                tp += 1
            elif not fired and not verdict:
                tn += 1
            elif fired and not verdict:
                fp += 1
            else:
                fn += 1
        alarms = ", ".join(false_alarms) if false_alarms else "none"
        print(f"{threshold:>7.2f} | {tp:>3} {tn:>3} {fp:>3} {fn:>3} | {n_verdict:>11} | {alarms[:36]:>36}")

    print("\nreference: the hand-written rule scored TP=62 TN=6 FP=2 FN=0 on all 70 verdict episodes")
    print("           (its 2 FPs, ep5 and ep8, were confirmed real grasps that failed at placement)")

    n_deliberate = sum(1 for e in scored if episode_map.get(e, {}).get("deliberate_negative"))
    n_verdicts = sum(1 for e in scored if episode_map.get(e, {}).get("verdict") is not None
                     and not episode_map.get(e, {}).get("deliberate_negative"))
    print(f"\nscored {n_verdicts} episodes carrying a human verdict and "
          f"{n_deliberate} deliberate-negative episodes")
    if not args.include_train:
        print("all held out from training -- pass --include-train to also score seen episodes")


if __name__ == "__main__":
    main()
