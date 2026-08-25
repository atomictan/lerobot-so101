# HIL-SERL config notes — SO-101 tube grasp

draccus rejects unknown keys, so these annotations cannot live in the JSON.
`hilserl_env_only.json` tests the environment alone; `hilserl_grab_tube.json` drives the
actor/learner. They differ in the `dataset` block: gym_manipulator's DatasetConfig requires
`task`, the training one does not.

## overview

HIL-SERL / SAC config for the SO-101 tube grasp. Every value here was derived or measured earlier in this project; see the notes beside each block. Launch: learner first, then actor, both with --config_path pointing here.

## dataset

The offline half of the 50/50 mixer. 36 orientation-matched demos, FK-converted to EE deltas, decimated to 10 Hz, truncated to the in-bounds approach ending at the grasp. 2,428 transitions, one terminal reward each.

## env.no_type

HILSerlRobotEnvConfig rejects a `type` key -- do not add one.

## env.teleop

Intervention device. The LEADER ARM CANNOT be used: AddTeleopEventsAsInfoStep calls _check_teleop_with_events, which requires get_teleop_events(), and SOLeader does not implement it. Only GamepadTeleop and KeyboardEndEffectorTeleop do. Keys: arrows/shift/ctrl nudge the end effector and set is_intervention while held; s = mark success, q = terminate, r = terminate and re-record.

## env.processor.observation

add_ee_pose_to_observation REPLACES the 6 joint angles with 7 EE values [x,y,z,wx,wy,wz,gripper_pos], putting state in the same coordinate frame as the action. The offline buffer was built to match exactly.

## env.processor.images

MUST match what the reward classifier trained on. The side crop keeps the full workspace column but only the left third where the tube sits; the wrist view is already almost all workspace so it is resized only. 128x128 is not a preference -- SpatialLearnedEmbeddings(4,4) is hardcoded and ResNet10 downsamples 32x.

## env.processor.ik

URDF validated against a printed grid: scale 1.008, distortion <=2.5% over 20 cm. Bounds confine the arm to the left half of the workspace where the tube is; the box and its rim sit at y > -0.01 and are therefore physically unreachable, which is what makes 'nothing else is clampable' true by construction rather than by habit. The z floor stops exploration driving into the table. Step size 0.02 m/unit matches both the HIL-SERL guide and the measured demo displacement (0.06% of actions clip).

## env.processor.gripper

max_gripper_pos=40 matters: with speed_factor=1.0 a single discrete action saturates to the clip limits, so the default 100 would open the jaws to ~98 -- far outside the 6.8-37.9 range the critic and the reward classifier ever saw. Verified on hardware: 0=open, 1=stay, 2=close (the library docstring says the opposite, which is wrong for this arm). The penalty discourages toggling open<->closed, not gripping.

## env.processor.reset

Reset pose chosen deliberately, then solved by IK: centred over the 36 demos' grasp centroid, 59 mm above the table, holding the reference wrist orientation to 1.6 deg. Margins to the bounds are 152/104/89 mm. terminate_on_success ends the episode at the first classifier positive -- which is why the offline demos were truncated there too.

## env.processor.reward

Checkpoint 006600 of reward_classifier_v2: 36 TP / 1 TN / 1 FP / 0 FN on 38 held-out human-verdict episodes, and 0 false alarms across all 10 held-out deliberate-negative episodes. success_threshold 0.7 was picked from the threshold sweep, not the 0.5 default.

## policy

storage_device stays cpu: the online buffer at 128x128 x 2 cameras is ~0.39 MB/transition, so 50k transitions is ~20 GB -- well past the 16 GB card but trivial against 187 GB of RAM. offline capacity just needs to hold the 2,428 demo transitions.
