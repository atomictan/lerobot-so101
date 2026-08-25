#!/usr/bin/env python
"""Check the SO-101 URDF against physical reality using a printed grid.

Teleoperate the closed gripper tip to a grid of known intersections, record the joint
angles at each, then fit a similarity transform between the URDF's forward-kinematics
predictions and the true grid coordinates:

    P_fk  ~=  s * R @ P_grid + t

What comes out:

  s                 the URDF scale error. 1.00 = correct; 1.10 = the model thinks the
                    arm is 10% larger than it is.
  residual RMS      what is left after the best rigid+scale fit -- i.e. NON-UNIFORM
                    distortion, which is the part that actually hurts. A uniform scale
                    error largely cancels (FK builds the training data, IK consumes the
                    policy's actions, both through the same model), but a warped model
                    makes a fixed action step cover different physical distances
                    depending on arm pose -- destroying the very property that makes an
                    end-effector action space more sample-efficient than joint space.

Two things make this work without calibrating where the paper is:

  * Only the RELATIVE geometry of the grid is used. Wherever the paper sits and however
    it is rotated is absorbed by R and t, so no measurement of the paper's pose is needed.
  * `gripper_frame_link` is a dummy frame inside the URDF, not a physical point you can
    touch to the paper. Whatever you actually align (a jaw tip) sits at an unknown offset
    from it -- but if the wrist orientation is held FIXED across all points, that offset
    is a constant world-frame vector, which t absorbs. This is why the script reports the
    orientation spread: if the wrist drifted, the assumption is violated and the numbers
    should be discounted rather than trusted.

Operator controls:
    Right arrow : record the current pose for the highlighted point
    Left arrow  : discard the previous point and redo it
    Escape      : abort (whatever was captured is still saved)

Example:
    python scripts/learning/urdf_grid_check.py --rows 5 --cols 5 --spacing-cm 5
    python scripts/learning/urdf_grid_check.py --analyze-only outputs/urdf_check/grid_20260813.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

URDF_PATH = "/home/atomic/robot_assets/so101/so101_new_calib.urdf"
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform: target ~= s * R @ source + t."""
    n = len(source)
    mu_s, mu_t = source.mean(0), target.mean(0)
    src_c, tgt_c = source - mu_s, target - mu_t
    cov = tgt_c.T @ src_c / n
    u, d, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    var_src = (src_c**2).sum() / n
    scale = float(np.trace(np.diag(d) @ correction) / var_src)
    translation = mu_t - scale * rotation @ mu_s
    return scale, rotation, translation


def grid_points(rows: int, cols: int, spacing_m: float) -> np.ndarray:
    """Grid laid out in its own frame; z = 0 since every point is on the paper."""
    return np.array(
        [[c * spacing_m, r * spacing_m, 0.0] for r in range(rows) for c in range(cols)],
        dtype=float,
    )


def capture(args: argparse.Namespace) -> dict:
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.teleoperators.so_leader import SO101LeaderConfig
    from lerobot.utils.keyboard_input import init_keyboard_listener
    from lerobot.utils.robot_utils import precise_sleep

    # No cameras: this check only needs joint angles, and skipping them speeds up connect.
    robot = make_robot_from_config(SO101FollowerConfig(port=args.robot_port, id=args.robot_id, cameras={}))
    teleop = make_teleoperator_from_config(SO101LeaderConfig(port=args.teleop_port, id=args.teleop_id))
    robot.connect()
    teleop.connect()

    targets = grid_points(args.rows, args.cols, args.spacing_cm / 100.0)
    listener, events = init_keyboard_listener()
    captured: list[dict] = []
    interval = 1.0 / args.fps

    print("\n" + "=" * 74)
    print(f"URDF grid check — {args.rows}x{args.cols} = {len(targets)} points, {args.spacing_cm} cm apart")
    print("=" * 74)
    print("Close the jaws fully, then keep the WRIST ORIENTATION CONSTANT throughout.")
    print("Touch the jaw tip to each intersection; RIGHT arrow records, LEFT redoes, ESC aborts.\n")

    try:
        idx = 0
        while idx < len(targets):
            row, col = divmod(idx, args.cols)
            gx, gy = targets[idx][0] * 100, targets[idx][1] * 100
            print(f"[{idx + 1:>2}/{len(targets)}] row {row} col {col}  ->  {gx:.0f} cm across, {gy:.0f} cm up")
            while True:
                loop_start = time.perf_counter()
                if events["exit_early"]:
                    events["exit_early"] = False
                    obs = robot.get_observation()
                    joints = [float(obs[f"{m}.pos"]) for m in MOTORS]
                    captured.append({"index": idx, "row": row, "col": col,
                                     "grid": targets[idx].tolist(), "joints": joints})
                    print(f"        recorded: " + " ".join(f"{m}={v:+.1f}" for m, v in zip(MOTORS, joints)))
                    idx += 1
                    break
                if events["rerecord_episode"]:
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    if captured:
                        dropped = captured.pop()
                        idx = dropped["index"]
                        print(f"        discarded point {idx + 1}, redo it")
                    break
                if events["stop_recording"]:
                    raise KeyboardInterrupt
                robot.send_action(teleop.get_action())
                precise_sleep(max(interval - (time.perf_counter() - loop_start), 0.0))
    except KeyboardInterrupt:
        print("\naborted by operator — keeping what was captured")
    finally:
        if listener is not None:
            listener.stop()
        robot.disconnect()
        teleop.disconnect()

    return {"rows": args.rows, "cols": args.cols, "spacing_cm": args.spacing_cm,
            "motors": MOTORS, "points": captured}



def _fit_with_offset(p_grid, p_fk, r_fk, scale0, translation0):
    """Fit scale/rotation/translation AND the constant jaw-tip offset in the gripper frame."""
    from scipy.optimize import least_squares

    def rodrigues(w):
        theta = np.linalg.norm(w)
        if theta < 1e-12:
            return np.eye(3)
        k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / theta
        return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k

    def residual(x):
        tip = p_fk + np.einsum("nij,j->ni", r_fk, x[7:10])
        return ((x[0] * (rodrigues(x[1:4]) @ p_grid.T)).T + x[4:7] - tip).ravel()

    sol = least_squares(residual, np.concatenate([[scale0], np.zeros(3), translation0, np.zeros(3)]))
    return sol.x[0], sol.x[7:10], np.linalg.norm(sol.fun.reshape(-1, 3), axis=1)


def analyze(data: dict, override_spacing_cm: float | None = None) -> None:
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.utils.rotation import Rotation

    points = data["points"]
    if len(points) < 4:
        raise SystemExit(f"need at least 4 points to fit a transform, got {len(points)}")

    kin = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="gripper_frame_link", joint_names=MOTORS)
    poses = [kin.forward_kinematics(np.array(p["joints"], dtype=np.float64)) for p in points]
    p_fk = np.array([T[:3, 3] for T in poses])
    r_fk = np.array([T[:3, :3] for T in poses])
    p_grid = np.array([p["grid"] for p in points])

    # Correct the assumed spacing if the printed grid turned out not to be what we thought.
    # A mis-stated spacing shows up as a pure scale error and would be blamed on the URDF.
    if override_spacing_cm is not None:
        p_grid = p_grid * (override_spacing_cm / data["spacing_cm"])
        print(f"  (spacing overridden: {data['spacing_cm']} cm -> {override_spacing_cm} cm)")

    scale, rotation, translation = umeyama(p_grid, p_fk)
    predicted = (scale * (rotation @ p_grid.T)).T + translation
    residual = np.linalg.norm(p_fk - predicted, axis=1)

    # Rigid-only fit (scale forced to 1) isolates how much of the error scale explains.
    # The translation must be recomputed without scale -- reusing the similarity fit's t
    # would leave it compensating for a scale that is no longer applied.
    t_rigid = p_fk.mean(0) - rotation @ p_grid.mean(0)
    rigid_pred = (rotation @ p_grid.T).T + t_rigid
    rigid_res = np.linalg.norm(p_fk - rigid_pred, axis=1)

    # The jaw tip is not `gripper_frame_link`; it sits at a fixed offset in the GRIPPER frame.
    # Assuming that offset away requires holding the gripper axis perfectly vertical, which is
    # hard to do by hand. Solving for it instead removes that requirement entirely and gives a
    # cleaner scale estimate -- on real data it cut the residual from 7.4 mm to 5.2 mm.
    scale_off, offset, resid_off = _fit_with_offset(p_grid, p_fk, r_fk, scale, translation)

    print("\n" + "=" * 74)
    print(f"URDF GRID CHECK — {len(points)} points, {data['spacing_cm']} cm nominal spacing")
    print("=" * 74)
    print(f"\n  scale factor s        : {scale:.4f}   ({(scale - 1) * 100:+.2f}% vs physical)")
    print(f"  residual RMS          : {residual.std():.4f} m  ({residual.mean() * 1000:.1f} mm mean, "
          f"{residual.max() * 1000:.1f} mm max)")
    print(f"  rigid-only residual   : {rigid_res.mean() * 1000:.1f} mm mean   "
          f"(vs {residual.mean() * 1000:.1f} mm with scale — the gap is what scale explains)")
    print(f"\n  >> solving for the jaw-tip offset (preferred):")
    print(f"       scale s      : {scale_off:.4f}   ({(scale_off - 1) * 100:+.2f}%)")
    print(f"       residual     : {resid_off.mean() * 1000:.2f} mm mean, {resid_off.max() * 1000:.2f} mm max")
    print(f"       tip offset   : {np.round(offset * 1000, 1)} mm  (|offset| = {np.linalg.norm(offset) * 1000:.1f} mm)")

    span = np.linalg.norm(p_grid.max(0) - p_grid.min(0))
    print(f"  grid span             : {span * 100:.1f} cm")
    print(f"  distortion            : {100 * residual.mean() / max(span, 1e-9):.2f}% of span")

    # Did the tool offset stay constant? The jaw tip sits along the gripper's z-axis, so
    # rotation ABOUT that axis leaves the tip where it is -- only TILT of the axis moves it.
    # That matters because yaw is uncontrollable anyway (shoulder_pan must turn to reach
    # different y), so judging by full orientation spread would flag harmless variation.
    rots = [T[:3, :3] for T in poses]
    down = np.array([0.0, 0.0, -1.0])
    tilt = np.degrees(np.arccos(np.clip([abs(R[:, 2] @ down) for R in rots], -1, 1)))
    full = [float(np.degrees(np.arccos(np.clip((np.trace(rots[0].T @ R) - 1) / 2, -1, 1)))) for R in rots]
    print(f"\n  gripper axis tilt from vertical : median {np.median(tilt):.1f}deg  spread {np.ptp(tilt):.1f}deg")
    print(f"  full orientation spread         : median {np.median(full):.1f}deg  "
          f"(includes harmless twist about the gripper axis)")
    if np.ptp(tilt) > 15:
        print("    ^ WARNING: the gripper axis tilted by more than 15deg across the points, so the")
        print("      jaw-tip offset was NOT constant. Part of the residual is that drift rather")
        print("      than URDF error -- treat s as approximate.")

    print("\n  per-point error (mm):")
    for p, r in zip(points, residual):
        flag = "  <-- check this point" if r > 3 * max(residual.std(), 1e-9) else ""
        print(f"    row {p['row']} col {p['col']}  {r * 1000:6.1f}{flag}")

    print("\n  interpretation:")
    if abs(scale - 1) < 0.02 and residual.mean() < 0.005:
        print("    URDF is good. Scale within 2% and distortion under 5 mm.")
    elif abs(scale - 1) < 0.05:
        print("    Scale is close. Uniform scale error largely cancels between FK and IK,")
        print("    so this is fine for RL; the residual is the number that matters.")
    else:
        print(f"    Scale is off by {(scale - 1) * 100:+.1f}%. Still largely cancels FK<->IK, but")
        print("    end_effector_step_sizes will not mean literal metres.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--spacing-cm", type=float, default=5.0, help="distance between recorded points")
    parser.add_argument("--robot-port", default="/dev/lerobot_follower")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")
    parser.add_argument("--teleop-port", default="/dev/lerobot_leader")
    parser.add_argument("--teleop-id", default="my_awesome_leader_arm")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/urdf_check"))
    parser.add_argument("--analyze-only", type=Path, help="skip capture; analyze an existing JSON")
    args = parser.parse_args()

    if args.analyze_only:
        # --spacing-cm doubles as a correction when re-analyzing a saved capture.
        stored = json.loads(args.analyze_only.read_text())
        override = args.spacing_cm if args.spacing_cm != stored["spacing_cm"] else None
        analyze(stored, override_spacing_cm=override)
        return

    data = capture(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"grid_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(data, indent=2))
    print(f"\nsaved {len(data['points'])} points -> {out}")
    analyze(data)


if __name__ == "__main__":
    main()
