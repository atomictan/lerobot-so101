# SO-101 Project Log: from imitation to reinforcement learning

**Project**: SO-101 arm, single task ("Grab the tube"), two cameras (wrist + side).
**Timeline**: July 8 – August 25, 2026.
**State**: imitation learning solved the task at 96.7%. The RL pipeline is validated in
simulation but has not yet been given enough real-robot interaction to converge. The open
question is a hardware-wear tradeoff, not a bug.

| Phase | Approach | Outcome |
|-------|----------|---------|
| 1 | ACT, real hardware | **96.7%** success (40% → 96.7% over 3 data cycles) |
| 2 | Simulation (PushT, ALOHA) | Policy–task fit understood; no new task solved (by design) |
| 3 | SAC / HIL-SERL | Pipeline validated in sim (100% success); real run under-trained |

Detailed Phase 1 writeup: [`GRASP_IMPROVEMENT_SUMMARY.md`](./GRASP_IMPROVEMENT_SUMMARY.md).
Config schema notes: [`hilserl_config_notes.md`](./hilserl_config_notes.md).

---

## 1. Phase 1 — Imitation learning on real hardware

An ACT policy went from 40% to 96.7% across three cycles of *evaluate → identify the
failure mode → record demonstrations targeting it → retrain*. Model architecture,
hyperparameters and hardware were unchanged throughout; only the data changed.

The decisive move was building [`evaluate_policy.py`](./evaluate_policy.py) — a harness
with keyboard start/stop and per-episode scoring — **before** trying to improve anything.
Casual testing had suggested ~70%; a fixed protocol measured 40%, because informal testing
samples the easy tube positions that the demos already covered.

**Lesson: build the measurement before the improvement.** Without a number you can
reproduce, "collect more data" is a guess. With one, every cycle targets a known weakness.

---

## 2. Phase 2 — Simulation, as a way of learning

Deliberately not aimed at solving a new task — this phase existed to build intuition.

| Policy | Task | Steps | Success |
|--------|------|-------|---------|
| ACT | PushT | 50k | 0/10 |
| Diffusion Policy | PushT | same data + budget | 1/10, ~2× ACT's coverage |
| ACT | ALOHA transfer-cube | 40k | 5/10 |

ACT scoring 0/10 on PushT looks like a training failure until you notice the loss curve is
decoupled from task performance. PushT is **multimodal** — many equally good ways to push
the block exist — and ACT's zero-latent inference averages incompatible modes into
something that does neither. The controlled A/B (changing only the policy) confirmed it,
and ACT on ALOHA transfer-cube — the task from the paper that introduced ACT — scored 5/10.

**Lesson: loss is not performance.** A policy can converge cleanly and still be
structurally incapable of a task. Changing one variable at a time is what separates "wrong
policy" from "not enough data".

Sim gotchas worth remembering: sim eval needs `--eval.use_async_envs=false` (forkserver
misses the `gym_pusht` registration), and old Hub policies predate the processor-pipeline
format and won't load.

---

## 3. Phase 3 — Reinforcement learning

### 3.1 The reward problem

Simulation hands you ground truth; a real arm does not know whether it is holding the tube.
Closing that gap took roughly three weeks and produced a vision-based reward classifier.

Labels came from a rule reading **motor signals only, no pixels**
([`label_reward.py`](./label_reward.py)), which auto-labelled 238 episodes:

```
action[grip] < min + 0.20*range   (per-episode, COMMANDED)
AND state[grip] >= 6              (absolute,   ACHIEVED)
AND run >= 60 frames              (5-frame gap bridging)
AND |pan - pan_at_grasp| < 20°
```

The non-obvious requirement: **the two gripper bounds must read different signals.** One
checks the commanded position, the other the achieved position. Using commanded for both
loses 73/118 demos; using achieved for both loses 10.

| Reward classifier | |
|---|---|
| Training data | 238 eps / 183,004 frames |
| Deliberate negatives recorded | 50 eps (3 sessions: close-on-nothing, wrong-object, near-miss) |
| Held-out result (`success_threshold=0.7`) | 36 TP / 1 TN / 1 FP / 0 FN |
| False positives on held-out negatives | 0 / 10 |
| Accepted episode error | ~2.6% |

The single false positive is a genuine grasp-then-slip — the tube really was between the
closing jaws for about a second. A single-frame classifier cannot see a temporal slip, and
no threshold or run-length filter removed it without costing twelve true detections.
Accepted rather than papered over.

Gotchas: `SpatialLearnedEmbeddings(4,4)` is hardcoded, so input **must** be 128×128, and
nothing in the training path crops or resizes — the crop is baked into the dataset
([`crop_dataset.py`](./crop_dataset.py)). `eval_loss` is useless at 8.8% positives (all-zeros
scores 91.2%); score at the episode level instead ([`eval_reward_classifier.py`](./eval_reward_classifier.py)).

### 3.2 Kinematics, and a false alarm worth remembering

RL needed end-effector control, so the SO-101 URDF was validated against a printed grid
([`urdf_grid_check.py`](./urdf_grid_check.py)) using a Umeyama similarity fit.

The first fit showed a **10% scale error** — alarming, and entirely fake. The printed grid
was 3.6 cm, not the assumed 4 cm. Corrected, the URDF measured **1.0082** (+0.8% scale),
distortion ≤5.15 mm over 20.4 cm.

**Lesson: your measuring instrument is also a hypothesis.** A grid-spacing error
masquerades *exactly* as a model scale error. When validation fails, the validator is a
suspect too.

placo gotchas: `forward_kinematics` takes **degrees**, rejects **float32** (LeRobotDataset
gives float32 — cast to float64), and `inverse_kinematics` does **one solver step per
call**; more than five iterations makes tracking worse through null-space drift.

### 3.3 The real-robot run

HIL-SERL running end-to-end: actor collecting at 10 Hz, learner optimizing at ~6.5 Hz and
pushing parameters back every ~2 s, human intervention by keyboard. Config:
[`hilserl_grab_tube.json`](./hilserl_grab_tube.json).

Over 20 interventions across 50 episodes, with a ~60% success rate — and then the question
that reframed everything: *is it actually learning?*

The metrics that would answer it, episodic reward and intervention rate, were being
computed by the actor and then discarded, because they only ever went to wandb, which was
disabled. Reconstructing them from the actor log and the saved replay buffer gave an
uncomfortable answer:

> **90–98% of every episode was human-driven.** The ~60% success rate measured the
> operator, not the policy. The one episode where the policy drove itself for 140 steps
> ended in failure.

At that intervention level the policy is not being taught, it is being replaced. It never
acts, so the critic never learns what the policy's *own* actions are worth — and the critic
is the actor's only teacher (`_compute_loss_actor` never touches `batch[ACTION]`; it samples
fresh actions from the policy and maximizes the critic's estimate of them).

### 3.4 What simulation settled in 90 minutes

Rather than keep grinding the real arm, the same pipeline — same SAC, same actor/learner,
same code — was pointed at `gym_hil/PandaPickCubeKeyboard-v0`
([`gym_hil_pickcube.json`](./gym_hil_pickcube.json)) and left running unattended with **zero
intervention**.

```
  block          success   ep_len   cum transitions
  ep    1-138      4.3%     93.0        12,831
  ep  139-276      3.6%     97.2        26,244     <- nothing happening
  ep  277-414     52.9%     58.8        34,358     <- it clicks
  ep  415-552    100.0%     11.8        35,984
  ep  553-690    100.0%     10.9        37,492
  ep  691-828    100.0%     10.4        38,925
  ep  829-966     97.8%     12.6        40,660
  ep  967-1104    99.3%     10.9        42,171
```

It solved the task completely. **This curve is the most valuable artifact the project has
produced** — the reference every future run gets compared against.

- **RL improves as a phase transition, not a gradient.** Nothing for 279 episodes
  (~26,000 transitions), then 4% → 100% in under 200 episodes. A run killed at episode 250
  looks identical to one that will never work.
- **Episode length is the earlier, sharper signal**: 92 → 10 steps, and it starts falling
  (98 → 72) *before* success starts rising.
- It works with no human because 30 demo episodes carry the reward signal — this is RLPD.
  Human intervention is a sample-efficiency accelerator, not a requirement.

**This reframes the real run.** It had collected **~2,600 online transitions**; sim showed
no signal at all until **~26,000**. The real run was not failing — it had not started. Do
not judge a real run on fewer than ~25k transitions.

### 3.5 Does intervening actually help? A controlled test

With a zero-intervention baseline in hand, the run was repeated with interventions.

| Run | Mean intervention | Transitions to sustained 90% | Speedup |
|-----|------------------|------------------------------|---------|
| 1 — hands off | 0% | 33,655 | — |
| 2 — with intervention | 12.9% | 30,718 | **1.10×** |

Essentially no speedup, against a predicted 2–5×. With one run per condition that is within
noise, but the *shape* of run 2 was informative: interventions were concentrated in the
first 150 episodes, then stopped, and the run went through the same exploration desert
anyway.

**Lesson: intervene late, not early.** When the policy is random, corrections mostly supply
examples of success — which the offline demo buffer already provides in half of every batch.
Intervention's unique contribution is correcting *policy-specific* failures, which requires
a policy good enough to fail in specific ways.

---

## 4. Sixteen fixes to the library itself

Running a less-travelled code path at depth surfaces things. Three were in the keyboard
intervention path alone; two were image-scale bugs that only appeared once checkpointing ran
for the first time. Committed in `8984b5ca`.

| Bug | Why it mattered |
|-----|-----------------|
| Released keys overwrote held keys | `get_action()` looped over every key ever pressed and assigned unconditionally, so a stale `down: False` zeroed a held `up`, with dict insertion order deciding the winner |
| `current_pressed.clear()` per poll | One step of control, then a ~500 ms dead zone |
| Reward-classifier images unnormalized at inference | Training ImageNet-normalized, inference did not; the frozen encoder made it matter |
| Buffer dump wrote floats to the image writer | Writer accepts float only in [0,1]; every frame failed and the dumped buffer had **no images**, making resume unusable |
| Resume path rescaled images differently | `make_dataset()` yields uint8 [0,255], a bare `LeRobotDataset` yields float32 [0,1] — a resumed run fed the encoder images **255× darker**, silently |
| Metrics computed then discarded | Episodic reward and intervention rate never surfaced unless wandb was enabled |
| EE jump raised instead of clamping | A 0.067 m > 0.05 m jump was killing runs mid-episode |
| Fixed 0.75 s reset regardless of distance | Now constant-speed |

**The dangerous ones raised nothing at all.** Verifying a round-trip — value in, value out,
unchanged — caught what error handling never would.

---

## 5. Lessons distilled

1. **Build the measurement before the improvement.** Phase 1's eval harness drove the
   40 → 96.7% climb. Phase 3 stalled partly because the equivalent measurement was being
   thrown away.
2. **RL fails and succeeds on a completely different timescale than imitation.** Patience is
   a technical requirement, not a temperament.
3. **Human help has an optimum, and it is lower than instinct suggests.** At 96%
   intervention the policy never acts, so the critic never learns what its own actions are
   worth.
4. **Simulation's value here was diagnostic, not developmental.** 90 minutes of unattended
   sim answered a question three weeks of real-robot work could not.
5. **Silent failures outnumber loud ones.** Of sixteen library bugs, the dangerous ones
   raised no error.

---

## 6. Where it stands

The RL pipeline is validated and the task is well understood. What remains is a genuine
engineering tradeoff, not a bug: reaching sim-scale interaction on physical hardware means
roughly **34,000 transitions** — about an hour of continuous motion plus resets, on servos
where the gripper motor (ID 6) has already dropped off the bus once from heat.

Options, each with a real cost:

- **Run it physically** and accept the wear.
- **Build an SO-101 simulation** and inherit the vision sim-to-real gap (rendered images vs.
  webcam feeds is the hard part; contact physics for a compliant printed gripper is the
  other).
- **Record more demonstrations.** The real offline buffer holds 19 episodes / 811
  transitions versus the 30 that carried the sim run. A thicker buffer shortens the
  exploration desert with no RL wall-clock on the arm at all.

Known-open items: gripper `speed_factor` is hardcoded to 1.0, so every gripper command
saturates end-to-end at 10 Hz. Lowering it to ~0.15 requires a matching offline-buffer
rebuild, since the buffer's gripper actions assume saturation.

---

## 7. File index

| File | Purpose |
|------|---------|
| [`evaluate_policy.py`](./evaluate_policy.py) | Eval harness: keyboard start/stop, per-episode scoring, failure collection |
| [`label_reward.py`](./label_reward.py) | Motor-signal labelling rule → `next.reward` |
| [`crop_dataset.py`](./crop_dataset.py) | Crop/scale to the 128×128 the classifier requires |
| [`eval_reward_classifier.py`](./eval_reward_classifier.py) | Per-frame P/R sweep + episode-level scoring vs. human verdicts |
| [`build_offline_buffer.py`](./build_offline_buffer.py) | Demos → bounds-consistent offline replay buffer |
| [`urdf_grid_check.py`](./urdf_grid_check.py) | Validate URDF scale against a printed grid (Umeyama fit) |
| [`gripper_sign_check.py`](./gripper_sign_check.py) | Hardware check for gripper action polarity |
| [`hilserl_grab_tube.json`](./hilserl_grab_tube.json) | Real-arm actor/learner config |
| [`hilserl_env_only.json`](./hilserl_env_only.json) | Real-arm env-only test config |
| [`gym_hil_pickcube.json`](./gym_hil_pickcube.json) | Sim config that validated the RL pipeline |
| [`GRASP_IMPROVEMENT_SUMMARY.md`](./GRASP_IMPROVEMENT_SUMMARY.md) | Phase 1 detail: 40% → 96.7% |
| [`hilserl_config_notes.md`](./hilserl_config_notes.md) | The two config schemas and their incompatibilities |

Two non-obvious build notes: episodes **must** be truncated at the first positive reward
(HIL-SERL terminates on success), and the terminal frame **must** be kept with a null action
or the buffer's `reward[i]` / last-frame-`done` convention discards the only reward-1
transition. Build datasets via `LeRobotDataset.create` / `add_frame` / `save_episode` +
`finalize()` inside `VideoEncodingManager`; hand-rolling metadata breaks.
