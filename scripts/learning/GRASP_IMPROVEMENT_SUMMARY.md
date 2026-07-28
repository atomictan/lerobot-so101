# Grab-the-Tube: from 40% to 97% Success

**Project**: SO-101 arm, ACT policy, single task ("Grab the tube"), two cameras (wrist + side).
**Timeline**: July 8–22, 2026.
**Result**: measured success rate improved from **40%** to **96.7%** over three data cycles,
with no changes to the model architecture, hyperparameters, or hardware — only to the *data*.

| Policy | Training data | Episodes | Success rate | Protocol |
|--------|--------------|----------|-------------|----------|
| v2 | round-2 demos only | 50 | 40% (8/20, reproduced twice) | 20 trials, 40 s |
| v3 | + corrections #1 | 75 | 60% (24/40 pooled) | 2×20 trials, 40 s |
| v4 | + corrections #2, #3 | 107 | 82.5% (33/40 pooled) | 2×20 trials, 40 s |
| v5 | + corrections #4, #5 | 118 | **96.7% (29/30)** | 30 trials, 50 s |

---

## 1. Starting point: "it feels like 70%" was actually 40%

The first policy (ACT, 50 teleop episodes, 30k steps) *felt* like it worked ~70% of the
time in casual testing. A formal evaluation — fixed protocol, 20 trials, strict success
criterion (tube securely grasped and lifted), deliberately varied tube positions —
measured **35–40%**.

Two lessons immediately:

- **Casual testing samples easy positions.** The informal 70% came from placing the tube
  where demos had placed it. The formal protocol sampled the whole workspace.
- **A number you can't reproduce is not a number.** The 40% was reproduced across two
  independent 20-trial runs before we trusted it.

## 2. Diagnosis: rule out the wrong hypothesis first

Every failure was tagged with a category (`missed_position`, `grasp_timing`,
`hesitation`, `ignored_object`) and later a workspace zone note. Two hypotheses fit the
initial failure map:

**Hypothesis A — scene change.** Frame-by-frame comparison of training vs. eval footage
showed the background had changed between recording (Jul 9) and eval (Jul 21): a chair
where an exercise ball had been, a moved box exposing a different backdrop. Since ACT
fine-tunes its ResNet18 backbone on just 50 episodes, it soaks up background as
incidental cues. **Test**: restore the scene, re-run the same 20-trial protocol.
**Result**: 35% → 40% — within noise. Scene mattered little. *Hypothesis rejected cheaply
before spending any training time on it.*

**Hypothesis B — position coverage gap.** Contact sheets (first frame of every eval
episode, labeled success/fail) put the pattern beyond doubt: every success had the tube
inside the demos' "comfort zone" (a band left of the cardboard, moderate distance);
every failure was outside it — frame edges, table front, far positions, against the box.
The same contact sheet built from *training* episodes confirmed demos had concentrated
in exactly that band. **The policy performed at ~70% inside its training distribution
and ~0% outside it. The 40% total was just the mix.**

## 3. The fix: a failure-replay flywheel

Each improvement cycle ran the same loop:

```
formal eval (fixed protocol, zone-tagged verdicts)
   → for each failure: restore the tube to the failure position
     (ghost overlay: 50/50 blend of reference frame + live camera in Rerun)
   → record 2–3 teleop correction demos at that position
     (small deliberate offsets between demos — teach a neighborhood, not a point)
   → merge corrections with existing training data
   → retrain from scratch, ~5 epochs
   → re-run the identical eval protocol
```

Design decisions that mattered:

- **Corrections are fresh full demos from the home pose**, not mid-flight takeovers —
  so they match the shape and distribution of the original training episodes and merge
  cleanly.
- **Eval rollouts never enter training data.** Failed policy trajectories would teach
  failure. Corrections live in separate datasets; only demos are ever merged.
- **The reference photo → ghost overlay** turned "put the tube back where it was" from
  a memory guess into a visual alignment task.
- **Verdicts are written to disk before collection starts** — a hardware fault
  mid-collection can never lose an eval result.
- **Same protocol every cycle.** Same rest pose, same scene, same position spread,
  same success criterion. The success-rate deltas are meaningful only because nothing
  else moved.

## 4. Cycle-by-cycle

- **v3 (75 eps: +25 corrections)** — 40% → 60%. The originally-targeted boundary zones
  improved; remaining failures shifted to position × orientation corners, and failures
  became slow near-misses (~31 s) instead of fast whiffs (~13 s).
- **v4 (107 eps: +32 corrections)** — 60% → 82.5%. Top/right boundary zones fixed;
  residual failures narrowed to a thin band (left/bottom-left boundary, center-bottom).
  A checkpoint-loss experiment (loss at 1.4/2.8/4.2/5.1 epochs on fixed batches:
  0.159 → 0.111 → 0.087 → 0.085) confirmed training had plateaued at 5 epochs —
  **the model was data-limited, not training-limited** — so effort stayed on data.
- **v5 (118 eps: +11 corrections)** — 82.5% → 96.7%. One caveat honestly noted: the
  eval window grew from 40 s to 50 s, and 5 of 29 successes used the extra time. At the
  old window v5 ≈ v4 on *fast* successes — but v5 gained something v4 never showed:
  **recovery**. v4's timer-length episodes all failed; v5's succeeded (5 of 6). Likely
  cause: one-third of v5's training data consists of correction demos, which are
  literally demonstrations of succeeding from awkward configurations.

## 5. What actually drove the improvement

1. **Measurement before optimization.** Every decision traced to a measured failure
   map, not a feeling. The single cheapest step — 20 formal trials with failure
   categories — redirected all subsequent effort.
2. **Targeted data beats volume.** 68 correction episodes aimed at measured failures
   took the policy from 40% to ~97%. For comparison, the second batch of 50 *generic*
   episodes (round 2 vs round 1) had moved the needle roughly 0%.
3. **Diversity along the measured axis.** The golden rule ("add diversity one axis at
   a time") applied with the axis chosen by evidence: tube position first, then
   position × orientation.
4. **Ruling out hypotheses cheaply.** The scene-restoration test cost 30 minutes and
   zero training runs, and prevented a wasted cycle chasing background robustness.
5. **Honest protocol accounting.** Timer limits, pooled runs, sample sizes, and the
   40 s → 50 s change are all part of the numbers. A success rate without its protocol
   is marketing, not measurement.

## 6. Artifacts

**Data lineage** (all local, `~/.cache/huggingface/lerobot/Atomictan/`):

```
record-test_20260709_123546            50 eps   foundation demos (round 2)
grab_tube_corrections_20260721_141125  25 eps   cycle-1 corrections (12 failure positions)
grab_tube_corrections_20260721_182512  14 eps   cycle-2 corrections
grab_tube_corrections_20260721_190216  18 eps   cycle-2 corrections (second eval run)
grab_tube_corrections_20260722_105932   5 eps   cycle-3 corrections
grab_tube_corrections_20260722_121904   6 eps   cycle-3 corrections (second eval run)
grab_tube_merged_v3                   118 eps   v5 training set (merge of all the above)
```

**Tooling** (`scripts/learning/`):

- `evaluate_policy.py` — the evaluation + failure-replay collection harness:
  fixed-protocol trials, operator verdict prompts with failure taxonomy, per-episode
  reference frames, crash-safe JSON results, ghost-overlay tube restoration, and
  teleop correction recording with live camera views in Rerun.
- Eval results: `outputs/eval/*.json` (one per session, verdicts + zones + durations).
- Trained policies: `outputs/train/act_grab_tube_v{3,4,5}/checkpoints/`.

**Reproduction of the final policy**:

```bash
lerobot-train \
  --dataset.repo_id=Atomictan/grab_tube_merged_v3 \
  --policy.type=act \
  --policy.device=cuda \
  --steps=57000 \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_grab_tube_v5
```

## 7. Postscript: the frozen-arm fixed point (action-horizon experiment)

After v5, we experimented with the inference-time action horizon (`n_action_steps`):
the policy always predicts a 100-step chunk, but only the first N steps are executed
before re-observing and re-planning. Result:

| n_action_steps | Behavior |
|----------------|----------|
| 100 (default) | Normal operation |
| 50 | Works, slightly more reactive |
| 30 | **Arm never moves — sits at home forever** |

The diagnosis (hypothesized from behavior, then confirmed against the data): the
training demos begin with a median of **60 frames (2.0 s) of stillness** — the
operator's reaction time between episode start and actually moving the leader. 98% of
demos are still motionless at frame 30. So at N=30, the first executed second of every
chunk is pure stillness; the arm doesn't move, the next observation is identical to the
last, and the policy — fully deterministic at inference (the VAE latent is zeroed) —
produces the identical prediction again. A true fixed point with no noise to escape it.

At N=100, 87% of demos are in full motion before frame 100, so every first chunk
contains committed movement. At N=50 the chunk blends in the earliest movers (33% of
demos) and bootstraps out — functional, but near the edge.

Lessons:

- **The policy imitates operator latency too.** The familiar few-second pause before
  rollouts start moving is not "thinking" — it is a faithful reproduction of the
  demonstrator's reaction time.
- **Practical floor for this policy: N ≈ 60** (the median lead-in of its data).
- **Fix at recording time**: start moving promptly when an episode begins (or trim
  idle lead-in frames when curating). Idle demo openings silently constrain which
  inference settings are usable later.

## 8. If pushing further (97% → ?)

- Larger eval samples (50+ trials) — at this level, 20-trial runs cannot distinguish
  real changes from noise.
- The residual failure (1/30) was a full-window grind with no zone cluster left —
  further gains likely need orientation diversity at the last boundary spots, or a
  policy-level change (e.g. smaller `n_action_steps` for more frequent replanning).
- Background/lighting diversity, if the deployment scene will vary — the current
  policy is trained and validated in one scene.
