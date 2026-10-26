# wm-eval

**A neutral referee for latent world-model planning.** Same model, same
episodes, pinned rules — so a success rate means one thing.

LeWorldModel's TwoRoom success rate has appeared in print as 8%, 14%, 84%,
87%, 94% and 97.4% for the *same released weights*
([Singh 2608.10145](https://arxiv.org/abs/2608.10145) for 8/14/84/94 and the
paper's ~87%; [TD-JEPA 2607.25337](https://arxiv.org/abs/2607.25337) for 97.4),
depending on how goals were constructed, how long the planner was allowed to
run, and which undocumented conventions a reproducer happened to follow.
Follow-up methods each compare against LeWM under their own setup. wm-eval
makes the setup an artifact:

- **Protocol cards** (`protocols/`) — every choice that can move a success
  rate (goal construction, budget, planner, normalization, seeds) in one YAML
  file, identified by the SHA-256 of its canonical form.
- **Episode packs** (`packs/`) — the exact start/goal pairs a protocol
  evaluates on, generated deterministically from the environment (no 3.4 GB
  dataset download) and verified by hash before every run.
- **Reproduction cards** (`results/`) — one JSON per run with per-episode
  outcomes, the protocol/pack/weights hashes and software versions, reported
  with Wilson 95% intervals, a random-policy floor and an oracle ceiling
  measured on the *same* episodes.

It builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
(pinned to a commit) for environments, solvers and models.

## Results

Released LeWM weights (`hf:quentinll/lewm-tworooms`, sha256 `566f223624ea…`),
200 episodes per protocol, Apple M2 (MPS):

| protocol | LeWM [95% CI] | random | expert | normalized |
|---|---|---|---|---:|
| `tworoom/repo-k25` | **82%** [77–87] | 24% | 100% | 0.77 |
| `tworoom/repo-k25-hist3` | **83%** [77–88] | 24% | 100% | 0.78 |
| `tworoom/k50-b100` | **61%** [54–67] | 12% | 100% | 0.55 |
| `tworoom/native-b100` | **34%** [28–41] | 8% | 100% | 0.28 |

*normalized* = (LeWM − random) / (expert − random), all three measured on the
same episodes. Full table with hashes and p-values:
[`results/README.md`](results/README.md); per-episode outcomes in
[`results/tworoom/`](results/tworoom/).

What the numbers say:

1. **The replica reproduces the moderate published figures, not the high
   ones.** 82% [77–87] agrees with Singh's 84% and is at the edge of the
   paper's ~87%. 94% and 97.4% fall outside the interval: under this
   protocol a 50-episode run would score ≥47/50 about 1% of the time, so
   those figures most likely come from a different setup.
2. **Goal construction alone moves the score by 48 points.** The same
   weights and planner score 82%, 61% or 34% depending on
   where goals come from (and the budget allowed to reach them). The random
   floor moves too (24% → 8%), so raw rates are not comparable across
   setups. Even normalized, the spread is 0.77 → 0.28.
3. **Feeding the predictor 3 context frames instead of 1 changes nothing
   measurable here.** 199/200 episodes have the same outcome (exact McNemar
   p = 1). That is expected once you look at the schedule: the planner
   replans every 25 env steps, so in a 50-step budget only the second plan
   ever sees the extra frames.

## Quick start

```bash
uv sync --extra run              # torch + stable-worldmodel (pinned)
export STABLEWM_HOME=~/.cache/wm-eval/swm

uv run wmeval pack protocols/tworoom/repo-k25.yaml          # verify the pack
uv run wmeval run  protocols/tworoom/repo-k25.yaml --policy random
uv run wmeval run  protocols/tworoom/repo-k25.yaml --policy expert
uv run wmeval run  protocols/tworoom/repo-k25.yaml \
    --policy hf:quentinll/lewm-tworooms --device mps         # or cpu / cuda
uv run wmeval report                                         # markdown table
```

`scripts/run_all.sh` reproduces every shipped result (90 min on an M2
MacBook with `mps`; LeWM is 13–34 min per protocol, baselines under a minute).

## Protocols shipped

| card | what changes vs `repo-k25` |
|---|---|
| `tworoom/repo-k25` | replica of the released LeWM eval config: goal 25 expert steps ahead, 50-step budget, CEM 300/30/30, horizon 5, frameskip 5 |
| `tworoom/repo-k25-hist3` | planner feeds 3 context frames (the predictor's trained `num_frames`) instead of 1 |
| `tworoom/k50-b100` | goals 50 expert steps ahead, 100-step budget |
| `tworoom/native-b100` | the environment's own start/target sampling, 100-step budget |

All cards use 200 episodes (the literature uses 50, whose 95% interval is
about ±13 points).

## How faithful is the replica?

- **Goals without the dataset.** `expert_offset` packs roll out the same noisy
  expert that generated the training data (`ExpertPolicy(action_noise=2.0,
  action_repeat_prob=0.05)`, 100-step episodes) and sample start rows
  uniformly, exactly as the dataset-driven evaluator samples rows.
- **Action normalization** is pinned to the StandardScaler fit on the released
  dataset (published by a third party). `scripts/check_norm.py` regenerates
  123k expert actions and recovers it within sampling error
  (std 0.8671/0.8695 vs pinned 0.8675/0.8686).
- **Goal images** are rendered by the environment with the agent placed at the
  goal; `scripts/check_goal_render.py` confirms they are pixel-identical to the
  observation at that position.
- **Random floor** under `repo-k25` lands at 23.5% (47/200), matching the ~26% that
  [le-wm#101](https://github.com/lucas-maes/le-wm/issues/101) reports for the
  released config.

## Safety

Checkpoints are untrusted input. Weights load with `torch.load(weights_only=True)`
(no pickle execution) and a downloaded `config.json` may only instantiate
architecture classes from an allowlist (`stable_worldmodel.wm.*`, `torch.nn.*`).
Released LeWM configs reference `stable_pretraining`'s ViT builder; wm-eval
rebuilds the identical `transformers.ViTModel` (strict state-dict load) so the
training stack is not needed.

## Known pitfalls this caught

- Released LeWM checkpoints do not load under `transformers>=5` (ViT
  state-dict keys were renamed); wm-eval pins `<5`.

## License

MIT. Built by [Omelas](https://omelas.tech).
