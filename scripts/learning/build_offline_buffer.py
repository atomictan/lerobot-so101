#!/usr/bin/env python
"""Convert teleop demos into the offline replay buffer HIL-SERL/SAC expects.

The demos were recorded for imitation learning: 30 fps, `action` = 6 joint positions,
`observation.state` = 6 joint positions. SAC needs something quite different:

    observation.state   7 values  [ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos]
    action              4 values  [delta_x, delta_y, delta_z, gripper]   at 10 Hz

`ReplayBuffer.from_lerobot_dataset` copies the `action` column verbatim, so demos are not
loadable as-is -- hence this script. Four transformations, each for a specific reason:

1. EPISODE FILTER. Orientation is frozen at the reset pose for the first run (SAC commands
   only position; the pipeline holds `desired[:3,:3] = ref[:3,:3]`). Demos where the wrist
   was rotated to a different angle describe motions the actor cannot reproduce, and worse,
   their recorded state change is partly rotational while the action records only
   translation -- teaching the critic dynamics that do not hold. Only demos whose grasp
   orientation is within `--orientation-tol` of the reference are kept.

2. FORWARD KINEMATICS. Joint angles become end-effector pose, so state and action share a
   coordinate frame. `add_ee_pose_to_observation=True` REPLACES joints with the EE pose at
   runtime, so the offline data must match exactly or the critic and actor see different
   input formats. (placo wants degrees and float64; LeRobotDataset hands out float32.)

3. DECIMATION to 10 Hz. A transition must span the same wall-clock interval offline and
   online, or the critic learns two incompatible dynamics models from one batch. Videos are
   re-encoded at 10 fps as well: LeRobotDataset resolves frames by seeking a timestamp, so
   dropping rows without re-timing the video would silently return the wrong images.

4. ACTIONS BY DIFFERENCING. `action[i]` is the motion that carries state `i` to state `i+1`,
   scaled by `end_effector_step_sizes` and clipped to [-1, 1]. The final frame of each
   episode has no successor and is dropped.

GRIPPER SIGN -- read this before trusting the output. `GripperVelocityToJoint` maps the
discrete action as `gripper_vel = -(a - 1) * clip_max`, so a=2 DECREASES the joint position.
Its comment calls a=0 "close", but that assumes position increases on close. On this arm the
opposite holds (open ~25-35, tube held ~10-13, closed on nothing ~1-3), so here:

    0 = open,  1 = stay,  2 = close

That is inverted relative to the code comment. It is a discrete right-or-wrong that no
amount of FK/IK cancellation fixes, and it can only be settled on hardware: command 2 and
watch the jaws. Verify before the first RL session.

Example:
    python scripts/learning/build_offline_buffer.py \
        --src Atomictan/grab_tube_reward_v2_128 \
        --dst Atomictan/grab_tube_offline_10hz
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

URDF_PATH = "/home/atomic/robot_assets/so101/so101_new_calib.urdf"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_IDX = 5
LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

# The env's `observation.state` is `agent_pos` = the 6 JOINT positions
# (VanillaObservationProcessorStep maps agent_pos -> observation.state).
# `add_ee_pose_to_observation` adds separate ee.* keys but does NOT rewrite
# observation.state, so the offline state must be joints to match the actor.
STATE_NAMES = list(MOTORS)
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "gripper"]

# Discrete gripper commands, in this arm's sign convention (see module docstring).
GRIPPER_OPEN, GRIPPER_STAY, GRIPPER_CLOSE = 0.0, 1.0, 2.0

# The safety box the RL policy is confined to; offline states must lie inside it too.
# Option A workspace. The original box (y>=-0.32, z<=0.15) was NOT reachable with the wrist
# orientation frozen -- z=0.15 is unreachable at every x, and past y~-0.24 the IK jumps between
# configurations, which showed up on hardware as the end effector oscillating. Measured
# reachability: 495/540 cells. z_max raised 0.05 -> 0.08 so the reset clears the tube:
# at z=0.05 the jaws sat too low to traverse over it.
BOUNDS_MIN = np.array([0.02, -0.23, -0.03])
BOUNDS_MAX = np.array([0.20, -0.03, 0.08])


def geodesic_deg(a: np.ndarray, b: np.ndarray) -> float:
    """True rotation angle between two orientations, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


def grasp_frame(action: np.ndarray, state: np.ndarray) -> int | None:
    """First frame of the sustained grasp, using the labelling rule's band + duration."""
    commanded, achieved = action[:, GRIPPER_IDX], state[:, GRIPPER_IDX]
    low, high = commanded.min(), commanded.max()
    if high - low < 1e-6:
        return None
    hits = np.flatnonzero((achieved >= 6.0) & (commanded < low + 0.20 * (high - low)))
    if hits.size == 0:
        return None
    runs = np.split(hits, np.where(np.diff(hits) > 5)[0] + 1)
    longest = max(runs, key=len)
    return int(longest[0]) if len(longest) >= 60 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="Atomictan/grab_tube_reward_v2_128")
    parser.add_argument("--dst", default="Atomictan/grab_tube_offline_10hz")
    parser.add_argument("--demo-episodes", type=int, default=118, help="leading episodes that are teleop demos")
    parser.add_argument("--reference-episode", type=int, default=26)
    parser.add_argument("--orientation-tol", type=float, default=40.0,
                        help="degrees from the reference grasp; 40 keeps 19 demos inside the "
                             "reachable box where 30 would keep only 11")
    parser.add_argument("--stride", type=int, default=3, help="30 fps / stride = control rate")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--step-size", type=float, default=0.02, help="metres per unit action")
    parser.add_argument("--gripper-deadband", type=float, default=1.0,
                        help="gripper position change below this counts as 'stay'")
    parser.add_argument("--task", default="Grab the tube")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.utils.rotation import Rotation

    src_root = LEROBOT_HOME / args.src
    dst_root = LEROBOT_HOME / args.dst
    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst_root} exists (pass --overwrite)")
        shutil.rmtree(dst_root)

    kin = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link", joint_names=MOTORS)
    source = LeRobotDataset(args.src, root=src_root)

    frames = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(src_root / "data" / "**" / "*.parquet"), recursive=True))])
    episodes = {
        int(ep): (
            np.stack(g.sort_values("frame_index")["action"].to_numpy()),
            np.stack(g.sort_values("frame_index")["observation.state"].to_numpy()).astype(np.float64),
            g.sort_values("frame_index")["next.reward"].to_numpy().astype(np.float32),
            g.sort_values("frame_index").index.to_numpy(),
        )
        for ep, g in frames.groupby("episode_index")
        if ep < args.demo_episodes
    }

    ref_action, ref_state, _, _ = episodes[args.reference_episode]
    ref_frame = grasp_frame(ref_action, ref_state)
    reference = kin.forward_kinematics(ref_state[ref_frame])[:3, :3]

    keep = []
    for ep, (action, state, _, _) in sorted(episodes.items()):
        f = grasp_frame(action, state)
        if f is None:
            continue
        if geodesic_deg(kin.forward_kinematics(state[f])[:3, :3], reference) <= args.orientation_tol:
            keep.append(ep)
    print(f"{len(keep)}/{len(episodes)} demos within {args.orientation_tol:.0f}deg of ep{args.reference_episode}'s grasp")

    image_keys = [k for k in source.meta.features if k.startswith("observation.images")]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
    }
    for key in image_keys:
        features[key] = {"dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channels"]}

    dataset = LeRobotDataset.create(
        repo_id=args.dst, fps=args.fps, features=features, root=dst_root,
        robot_type=source.meta.info.get("robot_type"), use_videos=True,
    )

    from lerobot.datasets import VideoEncodingManager

    clipped = 0
    total_actions = 0
    gripper_hist = {GRIPPER_OPEN: 0, GRIPPER_STAY: 0, GRIPPER_CLOSE: 0}
    roundtrip_errors = []

    skipped = []
    with VideoEncodingManager(dataset):
      for ep in keep:
          action_raw, state_raw, reward_raw, rows = episodes[ep]
          sel = np.arange(0, len(state_raw), args.stride)

          # Truncate at the FIRST positive frame. HIL-SERL terminates on success, so an online
          # episode is reset -> approach -> grasp -> done; carrying on through the transport to
          # the box would teach the critic about states the actor can never occupy. It also
          # removes the fast transport motion, where per-step displacement exceeds the action
          # range and clipping would corrupt the recorded dynamics (3.2% of steps vs 0.2%).
          positives = np.flatnonzero(reward_raw[sel] > 0)
          if positives.size == 0:
              skipped.append(ep)
              continue
          sel = sel[: int(positives[0]) + 1]
          if len(sel) < 2:
              skipped.append(ep)
              continue

          # Also truncate the START, at the first frame inside `end_effector_bounds`. Demos begin
          # at a home pose well outside the box the RL policy is confined to (y ~ +0.003 and
          # z ~ 0.184, against y <= -0.03 and z <= 0.15), so ~39% of frames described states the
          # actor can never occupy -- yet the critic would still learn values for them.
          # Keep the CONTIGUOUS in-bounds run that ends at the grasp. Truncating merely at the
          # first in-bounds frame is not enough: a trajectory can enter the box, leave it again,
          # and grasp outside -- which left 40% of states (and 13 of 32 terminal states) outside
          # the box the policy is confined to. If the grasp itself is unreachable, the whole
          # episode is unusable: its terminal reward sits somewhere the actor can never go.
          probe = np.array([kin.forward_kinematics(state_raw[i])[:3, 3] for i in sel])
          inside = np.all((probe >= BOUNDS_MIN) & (probe <= BOUNDS_MAX), axis=1)
          if not inside[-1]:
              skipped.append(ep)
              continue
          start = len(inside) - 1
          while start > 0 and inside[start - 1]:
              start -= 1
          if len(sel) - start < 2:
              skipped.append(ep)
              continue
          sel = sel[start:]

          poses = [kin.forward_kinematics(state_raw[i]) for i in sel]
          positions = np.array([T[:3, 3] for T in poses])
          rotvecs = np.array([Rotation.from_matrix(T[:3, :3]).as_rotvec() for T in poses])
          grippers = state_raw[sel, GRIPPER_IDX]

          # action[i] carries state i -> state i+1, so the last decimated frame has no action.
          deltas = np.diff(positions, axis=0) / args.step_size
          clipped += int((np.abs(deltas) > 1.0).sum())
          total_actions += deltas.size
          deltas = np.clip(deltas, -1.0, 1.0)

          d_grip = np.diff(grippers)
          gripper_cmd = np.full(len(d_grip), GRIPPER_STAY)
          gripper_cmd[d_grip < -args.gripper_deadband] = GRIPPER_CLOSE   # position falls => closing
          gripper_cmd[d_grip > +args.gripper_deadband] = GRIPPER_OPEN
          for v in gripper_cmd:
              gripper_hist[float(v)] += 1

          # Round-trip: replay the actions from the first pose. Nearly circular (the actions were
          # derived by differencing), so this only catches scaling/indexing/clipping -- not sign or
          # axis errors, which need the hardware.
          recon = positions[0] + np.cumsum(np.vstack([np.zeros(3), deltas * args.step_size]), axis=0)
          roundtrip_errors.append(np.linalg.norm(recon - positions, axis=1).max())

          # The terminal frame is KEPT with a null action. `from_lerobot_dataset` reads
          # reward[i] for transition i and marks the episode's last frame done=True, so dropping
          # it would discard the only reward-1 transition in the episode.
          for i in range(len(sel)):
              terminal = i == len(sel) - 1
              act = (np.array([0.0, 0.0, 0.0, GRIPPER_STAY]) if terminal
                     else np.concatenate([deltas[i], [gripper_cmd[i]]]))
              sample = source[int(rows[sel[i]])]
              frame = {
                  "observation.state": state_raw[sel[i]].astype(np.float32),
                  "action": act.astype(np.float32),
                  "next.reward": np.array([reward_raw[sel[i]]], dtype=np.float32),
                  "task": args.task,
              }
              for key in image_keys:
                  frame[key] = (sample[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
              dataset.add_frame(frame)
          dataset.save_episode()
          print(f"  ep{ep:>3}: {len(sel):>4} transitions  "
                f"({int((reward_raw[sel] > 0).sum()):>3} positive)  roundtrip max {roundtrip_errors[-1] * 1000:.3f} mm")

    dataset.finalize()

    print(f"\nwrote {args.dst}")
    print(f"  episodes           : {len(keep) - len(skipped)}" + (f"  (skipped {skipped})" if skipped else ""))
    print(f"  clipped actions    : {100 * clipped / max(total_actions, 1):.2f}% of components")
    total_grip = sum(gripper_hist.values())
    print(f"  gripper commands   : stay {100 * gripper_hist[GRIPPER_STAY] / total_grip:.1f}%  "
          f"close {100 * gripper_hist[GRIPPER_CLOSE] / total_grip:.1f}%  "
          f"open {100 * gripper_hist[GRIPPER_OPEN] / total_grip:.1f}%")
    print(f"  roundtrip error    : max {max(roundtrip_errors) * 1000:.3f} mm across all episodes")

    verify(args.dst, dst_root)


def verify(repo_id: str, root: Path) -> None:
    """Load it back the way the learner will, including the buffer's own conversion path."""
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import REWARD

    dataset = LeRobotDataset(repo_id, root=root)
    sample = dataset[0]
    print(f"\nverify: {len(dataset):,} frames, {dataset.meta.total_episodes} episodes, fps={dataset.meta.fps}")
    print(f"  observation.state {tuple(sample['observation.state'].shape)}  "
          f"action {tuple(sample['action'].shape)}  {REWARD} {float(sample[REWARD]):.0f}")
    for key in (k for k in sample if k.startswith("observation.images")):
        print(f"  {key}: {tuple(sample[key].shape)}")

    rewards = np.asarray(dataset.hf_dataset[REWARD], dtype=np.float32)
    actions = np.stack(dataset.hf_dataset["action"])
    print(f"  positives: {int((rewards > 0).sum()):,} / {len(rewards):,} ({100 * (rewards > 0).mean():.1f}%)")
    print(f"  action range: xyz [{actions[:, :3].min():+.3f}, {actions[:, :3].max():+.3f}]  "
          f"gripper values {sorted(set(actions[:, 3].tolist()))}")

    from lerobot.rl.buffer import ReplayBuffer

    state_keys = ["observation.state", *[k for k in sample if k.startswith("observation.images")]]
    buffer = ReplayBuffer.from_lerobot_dataset(
        dataset, device="cpu", state_keys=state_keys, storage_device="cpu",
        optimize_memory=True, capacity=len(dataset) + 8,
    )
    batch = buffer.sample(4)
    print(f"  ReplayBuffer.from_lerobot_dataset OK -> {len(buffer):,} transitions; "
          f"sampled batch keys {sorted(batch.keys())}")
    print(f"  batch action {tuple(batch['action'].shape)}  reward {tuple(batch['reward'].shape)}  "
          f"done {int(batch['done'].sum())}/4")
    assert torch.isfinite(batch["action"]).all(), "non-finite actions in the buffer"


if __name__ == "__main__":
    main()
