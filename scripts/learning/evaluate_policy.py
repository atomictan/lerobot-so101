#!/usr/bin/env python
"""Structured policy evaluation for the SO-101 grasp task.

Runs a trained policy for N episodes, guides the operator through a
consistent protocol, and records BOTH:
  - every frame of every episode into a LeRobotDataset (reviewable later), and
  - a per-episode verdict log (success/failure + category + notes) in a JSON
    file, with a final success-rate summary.

Operator controls during an episode:
    Right arrow / n : end the episode early (e.g. grasp clearly succeeded)
    Escape / q      : abort the whole session (in-progress data is kept)

After each episode the script asks for a verdict:
    s = success   f = failure   r = discard & redo this episode   q = quit

Example:
    python scripts/learning/evaluate_policy.py \
        --policy-path outputs/train/act_so101_test_round2/checkpoints/last/pretrained_model \
        --num-episodes 20 \
        --run-name round2_eval
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import time
from copy import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import teleop_smooth_move_to
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader import SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep

FAILURE_CATEGORIES = {
    "1": "missed_position",  # reached to the wrong place / tube position not covered
    "2": "grasp_timing",  # reached correctly but gripper closed early/late/weakly
    "3": "hesitation",  # hovered, oscillated, never committed
    "4": "ignored_object",  # did not orient toward the tube at all
    "5": "other",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", required=True, help="Path to pretrained_model directory")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--episode-time-s", type=float, default=40.0, help="Max seconds per episode")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task", default="Grab the tube")
    p.add_argument("--run-name", default=None, help="Short tag used in dataset/results names")
    p.add_argument("--robot-port", default="/dev/lerobot_follower")
    p.add_argument("--robot-id", default="my_awesome_follower_arm")
    p.add_argument("--device", default="cuda")
    p.add_argument("--results-dir", default="outputs/eval", help="Where the results JSON is written")
    p.add_argument(
        "--collect-on-failure",
        action="store_true",
        help="After each failed episode: guide tube restoration via a Rerun ghost overlay, "
        "then record teleop correction demos into a separate corrections dataset.",
    )
    p.add_argument("--teleop-port", default="/dev/lerobot_leader")
    p.add_argument("--teleop-id", default="my_awesome_leader_arm")
    p.add_argument("--correction-time-s", type=float, default=40.0, help="Max seconds per correction demo")
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Execute only the first N steps of each predicted chunk before replanning "
        "(default: the checkpoint's value, 100 for ACT). Lower = more frequent replanning.",
    )
    p.add_argument(
        "--manual-align",
        action="store_true",
        help="Never drive the leader arm with torque; the operator aligns it by hand. "
        "Use this if the leader's servos keep hitting protection faults.",
    )
    return p.parse_args()


def build_robot_config(args: argparse.Namespace) -> SO101FollowerConfig:
    return SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/lerobot_wrist_cam", width=480, height=640, fps=30, rotation=90
            ),
            "side": OpenCVCameraConfig(
                index_or_path="/dev/lerobot_side_cam", width=640, height=480, fps=30
            ),
        },
    )


def load_policy(args: argparse.Namespace):
    """Load policy + pre/post processors, mirroring rollout's context builder."""
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    if args.n_action_steps is not None:
        if args.n_action_steps > cfg.chunk_size:
            raise ValueError(f"--n-action-steps must be <= chunk_size ({cfg.chunk_size})")
        cfg.n_action_steps = args.n_action_steps
    policy_class = get_policy_class(cfg.type)
    policy = policy_class.from_pretrained(args.policy_path, config=cfg)
    policy = policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return policy, preprocessor, postprocessor


def smooth_move_to(robot, target: dict[str, float], duration_s: float = 2.0, fps: int = 50) -> None:
    """Linearly interpolate the arm to `target` so episodes start from the same pose."""
    obs = robot.get_observation()
    current = {k: v for k, v in obs.items() if k in target}
    steps = max(int(duration_s * fps), 1)
    for i in range(1, steps + 1):
        t = i / steps
        robot.send_action({k: current[k] * (1 - t) + target[k] * t for k in current})
        precise_sleep(1 / fps)


def run_episode(policy, preprocessor, postprocessor, robot, dataset, events, args) -> float:
    """Run the policy for one episode, recording every frame. Returns duration in seconds."""
    policy.reset()  # flush the action queue — never start on a stale plan
    device = torch.device(args.device)
    control_interval = 1.0 / args.fps
    start_t = time.perf_counter()
    timestamp = 0.0

    while timestamp < args.episode_time_s:
        loop_start = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break
        if events["stop_recording"]:
            break

        obs = robot.get_observation()
        obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)

        # Inference — same sequence as the rollout sync engine:
        observation = copy(obs_frame)
        with torch.inference_mode():
            observation = prepare_observation_for_inference(observation, device, args.task, robot.name)
            observation = preprocessor(observation)
            action = policy.select_action(observation)
            action = postprocessor(action)
        action_dict = make_robot_action(action.squeeze(0).cpu(), dataset.features)

        robot.send_action(action_dict)

        action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
        dataset.add_frame({**obs_frame, **action_frame, "task": args.task})

        dt_loop = time.perf_counter() - loop_start
        precise_sleep(max(control_interval - dt_loop, 0.0))
        timestamp = time.perf_counter() - start_t

    return timestamp


def ask_verdict(episode_num: int) -> dict | None:
    """Prompt the operator for the episode outcome. Returns None to redo, raises SystemExit on quit."""
    print("\n  Verdict for episode", episode_num)
    while True:
        v = input("  [s]uccess / [f]ailure / [r]edo episode / [q]uit session: ").strip().lower()
        if v == "s":
            notes = input("  Optional notes (Enter to skip): ").strip()
            return {"success": True, "category": None, "notes": notes}
        if v == "f":
            print("  Failure category:")
            for key, name in FAILURE_CATEGORIES.items():
                print(f"    {key} = {name}")
            cat = input("  Category [1-5]: ").strip()
            category = FAILURE_CATEGORIES.get(cat, "other")
            notes = input("  Optional notes (Enter to skip): ").strip()
            return {"success": False, "category": category, "notes": notes}
        if v == "r":
            return None
        if v == "q":
            raise SystemExit
        print("  Please answer s, f, r, or q.")


def write_results(path: Path, header: dict, episodes: list[dict]) -> None:
    """Rewrite the results file after every episode so a crash never loses the tally."""
    successes = sum(1 for e in episodes if e["success"])
    categories: dict[str, int] = {}
    for e in episodes:
        if not e["success"] and e["category"]:
            categories[e["category"]] = categories.get(e["category"], 0) + 1
    payload = {
        **header,
        "episodes_completed": len(episodes),
        "successes": successes,
        "success_rate": round(successes / len(episodes), 3) if episodes else None,
        "failure_categories": categories,
        "episodes": episodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def ghost_alignment(robot, ref_image: np.ndarray) -> None:
    """Stream a 50/50 blend of the reference frame and the live side camera to Rerun
    until the operator presses Enter. The operator moves the tube until the live tube
    coincides with the ghost."""
    import rerun as rr

    done = threading.Event()

    def _wait_enter():
        input("  Align the tube with the ghost in the Rerun window, then press Enter... ")
        done.set()

    t = threading.Thread(target=_wait_enter, daemon=True)
    t.start()
    ref = ref_image.astype(np.float32)
    while not done.is_set():
        obs = robot.get_observation()
        live = obs["side"].astype(np.float32)
        blend = (0.5 * ref + 0.5 * live).astype(np.uint8)
        rr.log("ghost_alignment/blend", rr.Image(blend))
        time.sleep(0.1)  # ~10 Hz is plenty for alignment


def record_teleop_demo(robot, teleop, dataset, args) -> float:
    """One teleop-driven demo episode, recorded frame by frame. Returns duration.

    Camera views are streamed to the Rerun viewer so the operator can teleoperate
    from the same viewpoints the policy will see.
    """
    import rerun as rr

    listener, events = init_keyboard_listener()
    control_interval = 1.0 / args.fps
    start_t = time.perf_counter()
    timestamp = 0.0
    try:
        while timestamp < args.correction_time_s:
            loop_start = time.perf_counter()
            if events["exit_early"] or events["stop_recording"]:
                break
            obs = robot.get_observation()
            action = teleop.get_action()
            robot.send_action(action)
            # .compress() sends JPEG instead of raw frames (~20x less bandwidth), which
            # keeps the Rerun server's 1 GiB history buffer from overflowing in seconds.
            rr.log("correction/wrist", rr.Image(obs["wrist"]).compress(jpeg_quality=75))
            rr.log("correction/side", rr.Image(obs["side"]).compress(jpeg_quality=75))
            obs_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
            action_frame = build_dataset_frame(dataset.features, action, prefix=ACTION)
            dataset.add_frame({**obs_frame, **action_frame, "task": args.task})
            precise_sleep(max(control_interval - (time.perf_counter() - loop_start), 0.0))
            timestamp = time.perf_counter() - start_t
    finally:
        if listener is not None:
            listener.stop()
    return timestamp


def force_disable_leader_torque(teleop) -> None:
    """Best-effort torque disable, one motor at a time.

    The bus-level disable_torque() iterates motors in ID order and aborts on the
    first faulted servo, leaving every later motor still torqued (a locked wrist
    with a faulted shoulder). Per-motor attempts free everything that can be freed.
    """
    for motor in teleop.bus.motors:
        try:
            teleop.bus.disable_torque(motor, num_retry=2)
        except Exception as e:  # noqa: BLE001
            print(f"  ! Could not disable torque on leader '{motor}': {e}")


def collect_corrections(robot, teleop, corrections_ds, home_position, ref_image, args) -> int:
    """Failure-replay collection: restore the tube via ghost overlay, then record
    one or more teleop demos from the home pose. Returns number of demos saved."""
    saved = 0
    print("\n  ── CORRECTION COLLECTION ──")
    print("  Restoring the failure scenario so you can demonstrate the correct grasp.")

    while True:
        # 1. Follower back to the start pose (it moved during the failure / previous demo).
        # 2. Leader aligned to it and torque-free.
        # 3. Tube restored onto its ghost — needed before EVERY demo, since each
        #    demo (and the failed attempt itself) displaces the tube.
        if args.manual_align:
            smooth_move_to(robot, home_position)
            force_disable_leader_torque(teleop)
            input("  Move the LEADER arm by hand to roughly match the follower's pose; "
                  "press Enter when aligned... ")
        else:
            input("  Let go of the leader arm, then press Enter (leader will move by itself)... ")
            smooth_move_to(robot, home_position)
            try:
                teleop_smooth_move_to(teleop, home_position, duration_s=3)
            except Exception as e:  # noqa: BLE001
                # Typically a servo protection fault (overload on shoulder_lift during
                # the assisted move). Fall back to manual alignment for this demo.
                print(f"  ! Assisted leader alignment failed ({e})")
                print("  ! If the leader feels stiff or a servo LED is blinking, power-cycle")
                print("  ! its DC supply after this session; consider --manual-align next run.")
            # Always free EVERY motor, even if some servo is faulted — otherwise the
            # motors after the faulted ID stay locked (unrotatable wrist).
            force_disable_leader_torque(teleop)
            input("  Check the leader moves freely, align it to the follower if needed, "
                  "then press Enter... ")

        ghost_alignment(robot, ref_image)

        input(f"  Grab the leader arm; press Enter to START the demo (max {args.correction_time_s:.0f}s, "
              "'n' to end early)... ")
        duration = record_teleop_demo(robot, teleop, corrections_ds, args)
        print(f"  Demo finished ({duration:.1f}s).")

        while True:
            v = input("  [s]ave this demo / [d]iscard it: ").strip().lower()
            if v in ("s", "d"):
                break
        if v == "s":
            corrections_ds.save_episode()
            saved += 1
        else:
            corrections_ds.clear_episode_buffer()

        v = input("  Record [a]nother demo at this position (nudge the tube a few cm for variety) "
                  "or [c]ontinue the eval: ").strip().lower()
        if v != "a":
            break
    print(f"  ── {saved} correction demo(s) saved ──")
    return saved


def main() -> None:
    args = parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or "eval"
    repo_id = f"Atomictan/rollout_eval_{run_name}_{stamp}"
    results_path = Path(args.results_dir) / f"{run_name}_{stamp}.json"

    print("=" * 62)
    print("POLICY EVALUATION SESSION")
    print("=" * 62)
    print(f"  Policy      : {args.policy_path}")
    print(f"  Episodes    : {args.num_episodes} (max {args.episode_time_s:.0f}s each)")
    print(f"  Dataset     : {repo_id}")
    print(f"  Results     : {results_path}")
    print()
    print("  OPERATOR PROTOCOL")
    print("  1. Between episodes, place the tube at the position asked for.")
    print("     Vary positions across the workspace — include hard ones.")
    print("  2. Do NOT touch the arm while an episode is running.")
    print("  3. Right arrow / 'n' ends an episode early once the outcome is clear.")
    print("  4. Judge success strictly: tube securely grasped and lifted = success.")
    print("     Anything else (dropped, pinched but slipped, pushed away) = failure.")
    print("=" * 62)

    policy, preprocessor, postprocessor = load_policy(args)
    print("Policy loaded.")

    robot = make_robot_from_config(build_robot_config(args))
    robot.connect()
    initial_obs = robot.get_observation()
    home_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}
    print("Robot connected. Current pose captured as the episode start pose.")

    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    dataset_features = {**obs_features, **action_features}
    dataset = LeRobotDataset.create(
        repo_id,
        args.fps,
        robot_type=robot.name,
        features=dataset_features,
        streaming_encoding=True,
        encoder_threads=2,
    )

    refframes_dir = Path(args.results_dir) / f"refframes_{run_name}_{stamp}"
    refframes_dir.mkdir(parents=True, exist_ok=True)

    teleop = None
    corrections_repo_id = None
    corrections_ds = None  # created lazily on first saved correction
    if args.collect_on_failure:
        corrections_repo_id = f"Atomictan/grab_tube_corrections_{stamp}"
        print("Connecting leader arm for correction collection...")
        teleop = make_teleoperator_from_config(
            SO101LeaderConfig(port=args.teleop_port, id=args.teleop_id)
        )
        teleop.connect()
        import rerun as rr

        rr.init("eval_corrections")
        rr.spawn(memory_limit="10%")
        print("Leader connected; Rerun viewer launched for ghost alignment.")

    def get_corrections_ds() -> LeRobotDataset:
        nonlocal corrections_ds
        if corrections_ds is None:
            corrections_ds = LeRobotDataset.create(
                corrections_repo_id,
                args.fps,
                robot_type=robot.name,
                features=dataset_features,
                streaming_encoding=True,
                encoder_threads=2,
            )
        return corrections_ds

    header = {
        "policy_path": args.policy_path,
        "task": args.task,
        "dataset_repo_id": repo_id,
        "num_episodes_planned": args.num_episodes,
        "episode_time_s": args.episode_time_s,
        "fps": args.fps,
        "started_at": stamp,
        "collect_on_failure": args.collect_on_failure,
        "corrections_repo_id": corrections_repo_id,
        "n_action_steps": args.n_action_steps,  # None = checkpoint default
    }
    episodes: list[dict] = []

    try:
        while len(episodes) < args.num_episodes:
            n = len(episodes) + 1
            print(f"\n{'─' * 62}")
            print(f"EPISODE {n}/{args.num_episodes}")
            input("  Place the tube, clear your hands, then press Enter to start... ")

            print("  Moving arm to start pose...")
            smooth_move_to(robot, home_position)

            # Reference frame: tube placed, arm at home. Used for the results record
            # and (in collect mode) as the ghost image for tube restoration.
            ref_image = robot.get_observation()["side"]
            ref_path = refframes_dir / f"ep_{n:02d}.png"
            Image.fromarray(ref_image).save(ref_path)

            # The key listener puts the terminal into raw/no-echo mode and eats
            # keystrokes, which breaks input(). So it only lives while the policy
            # runs, and is fully stopped before any text prompt below.
            listener, events = init_keyboard_listener()
            print(f"  Policy running (max {args.episode_time_s:.0f}s — Right arrow/'n' to end early)")
            duration = run_episode(policy, preprocessor, postprocessor, robot, dataset, events, args)
            if listener is not None:
                listener.stop()

            if events["stop_recording"]:  # Esc pressed mid-episode: abort session
                print("  Session aborted (Esc) — discarding the interrupted episode.")
                dataset.clear_episode_buffer()
                break

            try:
                verdict = ask_verdict(n)
            except SystemExit:
                dataset.clear_episode_buffer()
                break

            if verdict is None:  # redo
                print("  Discarding episode and repeating.")
                dataset.clear_episode_buffer()
                events["rerecord_episode"] = False
                continue

            dataset.save_episode()

            # Record the verdict BEFORE any correction collection, so a crash or
            # hardware fault during collection can never lose the eval result.
            entry = {
                "episode": n,
                "duration_s": round(duration, 2),
                **verdict,
                "ref_frame": str(ref_path),
                "corrections_saved": 0,
                "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
            }
            episodes.append(entry)
            write_results(results_path, header, episodes)

            if args.collect_on_failure and not verdict["success"]:
                v = input("  Collect correction demos for this failure? [y/N]: ").strip().lower()
                if v == "y":
                    entry["corrections_saved"] = collect_corrections(
                        robot, teleop, get_corrections_ds(), home_position, ref_image, args
                    )
                    write_results(results_path, header, episodes)
            done = sum(1 for e in episodes if e["success"])
            print(f"  Saved. Running tally: {done}/{len(episodes)} successes")
    finally:
        print("\nFinalizing dataset (this can take a moment)...")
        try:
            dataset.finalize()
        except Exception as e:  # noqa: BLE001
            print(f"  Dataset finalize issue (data up to last save is intact): {e}")
        if corrections_ds is not None:
            try:
                corrections_ds.finalize()
            except Exception as e:  # noqa: BLE001
                print(f"  Corrections finalize issue (saved episodes are intact): {e}")
        smooth_move_to(robot, home_position)
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"  Robot disconnect issue: {e}")
        if teleop is not None:
            try:
                teleop.disconnect()
            except Exception as e:  # noqa: BLE001
                print(f"  Leader disconnect issue (a faulted servo can cause this — "
                      f"power-cycle the leader's DC supply): {e}")

    print(f"\n{'=' * 62}")
    print("EVALUATION SUMMARY")
    if episodes:
        successes = sum(1 for e in episodes if e["success"])
        print(f"  Success rate : {successes}/{len(episodes)} = {successes / len(episodes):.0%}")
        categories: dict[str, int] = {}
        for e in episodes:
            if not e["success"] and e["category"]:
                categories[e["category"]] = categories.get(e["category"], 0) + 1
        if categories:
            print("  Failure breakdown:")
            for cat, count in sorted(categories.items(), key=lambda kv: -kv[1]):
                print(f"    {cat:<16} {count}")
    else:
        print("  No episodes completed.")
    print(f"  Full results : {results_path}")
    print(f"  Episode data : {repo_id}")
    print(f"  Ref frames   : {refframes_dir}")
    if corrections_ds is not None:
        total_corrections = sum(e.get("corrections_saved", 0) for e in episodes)
        print(f"  Corrections  : {corrections_repo_id} ({total_corrections} demos — training data)")
    print("=" * 62)


if __name__ == "__main__":
    main()
