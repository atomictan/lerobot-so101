#!/usr/bin/env python
"""Publish the grab-the-tube demonstration dataset to the Hugging Face Hub.

The dataset card is passed through `push_to_hub(**card_kwargs)` rather than
uploaded separately, because `push_to_hub` regenerates and overwrites README.md
on every call -- a card written beforehand would be silently replaced.

Uploads PRIVATE by default, so the card and video rendering can be checked on the
Hub before anything is exposed. Flip to public afterwards in the repo's Settings,
or pass --public here.

    python scripts/learning/push_dataset_to_hub.py --dry-run   # print card, upload nothing
    python scripts/learning/push_dataset_to_hub.py             # upload, private
    python scripts/learning/push_dataset_to_hub.py --public    # upload, public

Going public is irreversible in practice: the videos can be mirrored or cached
even if the repo is later deleted. Review the footage before that step.
"""

from __future__ import annotations

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = "Atomictan/grab_tube_merged_v3"

TAGS = ["so101", "so-arm101", "robotics", "manipulation", "grasping", "act", "real-robot"]

DESCRIPTION = """\
118 teleoperated demonstrations of an [SO-101](https://github.com/TheRobotStudio/SO-ARM100)
arm picking up a small tube from a mat and dropping it into a bin, recorded with two
cameras (wrist and side) at 30 fps.

**An ACT policy trained on this data reaches 96.7% success (29/30) on real hardware.**

That number is the point of this dataset. Plenty of demonstration sets exist; few state
what success rate the data actually produces, under what protocol, or how the data was
assembled to get there.

## Measured result

| Policy | Training episodes | Success rate | Protocol |
|--------|------------------|--------------|----------|
| v2 | 50 | 40% (8/20, reproduced twice) | 20 trials, 40 s |
| v3 | 75 | 60% (24/40 pooled) | 2x20 trials, 40 s |
| v4 | 107 | 82.5% (33/40 pooled) | 2x20 trials, 40 s |
| **v5 (this dataset)** | **118** | **96.7% (29/30)** | 30 trials, 50 s |

Trained with ACT for 57k steps. Success criterion: tube securely grasped and lifted,
with tube positions deliberately varied across the workspace.

## How it was built

Not one bulk recording session. The 118 episodes are 50 base demonstrations plus five
rounds of *targeted corrections*, each round recorded after evaluating the current policy
and identifying its specific failure mode. Architecture, hyperparameters and hardware were
unchanged throughout -- only the data changed.

The composition is the interesting part: episodes were added to cover workspace regions
and grasp situations where the policy was measurably failing, rather than to increase the
count.

## Usage

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("Atomictan/grab_tube_merged_v3")
```

Train an ACT policy on it:

```bash
lerobot-train \\
  --dataset.repo_id=Atomictan/grab_tube_merged_v3 \\
  --policy.type=act \\
  --output_dir=outputs/train/grab_tube \\
  --job_name=grab_tube \\
  --policy.device=cuda
```

It also works as an offline buffer for RL, though note it carries no reward column --
`ReplayBuffer.from_lerobot_dataset` requires `next.reward`, so you would need to label it
first (real demonstrations have no ground-truth success signal).

## Known limitations

- **Single task, single environment.** One tube, one mat, one lighting condition. Expect
  a policy trained on this to be sensitive to background changes; ACT fine-tunes its
  ResNet18 backbone on relatively few episodes and picks up incidental scene cues.
- **Tube positions are concentrated** where the correction cycles focused. Coverage is
  good across the reachable workspace but not uniform.
- **No reward labels.** Success/failure is not annotated per episode.
- **`robot_type` is recorded as `so_follower`**, from a local fork's naming, rather than
  the more standard `so101_follower`. The data itself is stock SO-101.

## Provenance and method

Recorded July 8-22, 2026 on a single hobbyist build, as part of learning robot ML
end-to-end.

The method worth copying is the measurement, not the data. Casual testing of the first
policy suggested roughly 70% success; a fixed protocol -- 20 trials, deliberately varied
tube positions, a strict success criterion -- measured 40%. Informal testing samples the
easy positions the demonstrations already covered.

From there each cycle was: evaluate under the fixed protocol, tag every failure with a
category and workspace zone, form a hypothesis about the dominant failure mode, and record
corrections targeting only that. One hypothesis (background/scene drift between recording
and evaluation) was tested by restoring the original scene and re-running the protocol; it
moved success 35% -> 40%, within noise, and was rejected before any training time was
spent on it. The position-coverage hypothesis survived, and drove the remaining cycles.

A fuller writeup will be published separately.
"""

STRUCTURE = """\
118 episodes / 95,305 frames / 30 fps / ~53 minutes per camera.

```
data/    parquet -- action, observation.state, timestamp, frame_index,
                    episode_index, index, task_index
videos/  observation.images.side   640x480, AV1   (9 files)
         observation.images.wrist  480x640, AV1   (7 files)
meta/    info.json, tasks.parquet, per-episode statistics
```

Both `action` and `observation.state` are 6-dimensional, in the same joint order:

```
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos,   wrist_roll.pos,    gripper.pos
```

`action` is the commanded position, `observation.state` the achieved position. The
difference between them is informative -- it is what a gripper-closure detector can be
built from when no ground-truth reward exists.

The two cameras chunk into different numbers of files (video files are split by size, not
episode count), so `side/file-004.mp4` and `wrist/file-004.mp4` cover different episodes.
`meta/episodes` holds the mapping.
"""

CITATION = """\
@misc{grab_tube_so101_2026,
  title  = {Grab the Tube: 118 SO-101 demonstrations reaching 96.7% with ACT},
  author = {Atomictan},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Atomictan/grab_tube_merged_v3}
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the card, upload nothing")
    parser.add_argument(
        "--public",
        action="store_true",
        help="upload public instead of private (irreversible in practice)",
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 78)
        print(f"TAGS: {TAGS}")
        print("=" * 78)
        print(DESCRIPTION)
        print("=" * 78)
        print(STRUCTURE)
        print("=" * 78)
        print(CITATION)
        print("=" * 78)
        print("Dry run -- nothing uploaded.")
        return

    ds = LeRobotDataset(args.repo_id)
    visibility = "PUBLIC" if args.public else "private"
    print(f"Uploading {ds.meta.total_episodes} episodes / {ds.meta.total_frames:,} frames...")
    print(f"Visibility: {visibility}.  ~2.5 GB, this will take a while.")

    ds.push_to_hub(
        tags=TAGS,
        license="apache-2.0",
        private=not args.public,
        upload_large_folder=True,
        dataset_description=DESCRIPTION,
        dataset_structure=STRUCTURE,
        citation_bibtex=CITATION,
        # No `url=` while the source repo is private -- it renders as "Homepage:" in the
        # card and would 404 for readers. Add it here once the repo is public and re-run;
        # push_to_hub regenerates the card, so updating it later is a one-command change.
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
