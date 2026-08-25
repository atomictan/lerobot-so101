#!/usr/bin/env python
"""Settle which discrete gripper action closes the jaws on THIS arm.

`GripperVelocityToJoint` maps the discrete command as:

    gripper_vel = -(a - 1) * clip_max        # a=0 -> +clip_max, a=1 -> 0, a=2 -> -clip_max
    gripper_pos = clip(current + gripper_vel * speed_factor, clip_min, clip_max)

and its docstring calls a=0 "close" -- but only because it assumes joint position INCREASES
on close. This arm is the opposite: the recorded demos show open ~25-35, tube held ~10-13,
closed on nothing ~1-3, i.e. position DECREASES on close. By that arithmetic a=2 should close.

That is a discrete right-or-wrong, and nothing offline can settle it: forward and inverse
kinematics are built from the same URDF, so any sign error there cancels and every offline
check passes regardless. If it is backwards, SAC learns inverted gripper control and the only
symptom is a policy that never succeeds.

Two parts:
  1. Direct position commands establish the ARM's convention (does a low value mean closed?).
  2. Discrete actions run through the REAL `GripperVelocityToJoint` step -- not a
     reimplementation -- establish the ACTION mapping, with the same speed_factor=1.0 and
     discrete_gripper=True that `gym_manipulator` uses.

Only the gripper is commanded; every other joint is held at its measured position.

Example:
    python scripts/learning/gripper_sign_check.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def hold_pose_with_gripper(robot, gripper_pos: float, seconds: float, fps: int = 30) -> float:
    """Command the current pose with only the gripper changed; return the settled position."""
    from lerobot.utils.robot_utils import precise_sleep

    obs = robot.get_observation()
    action = {f"{m}.pos": float(obs[f"{m}.pos"]) for m in MOTORS}
    action["gripper.pos"] = float(gripper_pos)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        loop = time.perf_counter()
        robot.send_action(action)
        precise_sleep(max(1.0 / fps - (time.perf_counter() - loop), 0.0))
    return float(robot.get_observation()["gripper.pos"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-port", default="/dev/lerobot_follower")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--max-gripper-pos", type=float, default=100.0,
                        help="clip_max passed to GripperVelocityToJoint (must match the RL config)")
    parser.add_argument("--settle-s", type=float, default=1.5)
    args = parser.parse_args()

    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.robot_kinematic_processor import GripperVelocityToJoint
    from lerobot.types import TransitionKey

    robot = make_robot_from_config(SO101FollowerConfig(port=args.robot_port, id=args.robot_id, cameras={}))
    robot.connect()

    try:
        start = float(robot.get_observation()["gripper.pos"])
        print(f"\nstarting gripper.pos = {start:.1f}")
        print("\n" + "=" * 70)
        print("PART 1 — the arm's convention: does a LOW value mean closed?")
        print("=" * 70)
        for target in (30.0, 5.0, 30.0):
            settled = hold_pose_with_gripper(robot, target, args.settle_s)
            print(f"  commanded gripper.pos = {target:5.1f}  ->  settled at {settled:5.1f}   "
                  f"WATCH THE JAWS")
            input("    press Enter after noting whether the jaws are OPEN or CLOSED... ")

        print("\n" + "=" * 70)
        print("PART 2 — the discrete action mapping, through the real processor step")
        print("=" * 70)
        step = GripperVelocityToJoint(
            speed_factor=1.0, clip_max=args.max_gripper_pos, discrete_gripper=True
        )
        results = {}
        for action_value in (0.0, 1.0, 2.0):
            # Reset to mid-open so both directions have room to show themselves.
            hold_pose_with_gripper(robot, 30.0, args.settle_s)
            before = float(robot.get_observation()["gripper.pos"])

            obs = robot.get_observation()
            step._current_transition = {TransitionKey.OBSERVATION: {f"{m}.pos": float(obs[f"{m}.pos"]) for m in MOTORS}}
            commanded = step.action({"ee.gripper_vel": action_value})["ee.gripper_pos"]

            after = hold_pose_with_gripper(robot, commanded, args.settle_s)
            results[action_value] = (before, commanded, after)
            verdict = "CLOSED" if after < before - 2 else ("OPENED" if after > before + 2 else "unchanged")
            print(f"  action {action_value:.0f}: step commanded {commanded:6.1f}   "
                  f"gripper {before:5.1f} -> {after:5.1f}   => {verdict}")
            input("    press Enter after confirming visually... ")

        print("\n" + "=" * 70)
        closing = [a for a, (b, _, af) in results.items() if af < b - 2]
        opening = [a for a, (b, _, af) in results.items() if af > b + 2]
        print(f"  CLOSES the jaws : action {closing}")
        print(f"  OPENS  the jaws : action {opening}")
        if closing == [2.0] and opening == [0.0]:
            print("\n  CONFIRMED: 0 = open, 1 = stay, 2 = close  (as build_offline_buffer.py assumes)")
            print("  No change needed.")
        elif closing == [0.0] and opening == [2.0]:
            print("\n  INVERTED vs what build_offline_buffer.py assumed!")
            print("  Swap GRIPPER_OPEN/GRIPPER_CLOSE in build_offline_buffer.py and rebuild the buffer.")
        else:
            print("\n  INCONCLUSIVE — check that the jaws were free to move and nothing was blocking them.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
