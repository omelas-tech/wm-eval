"""Cross-check the pinned TwoRoom action normalization.

The protocol cards pin the action mean/std that a third party fit on the
released 3.4 GB TwoRoom dataset. This script regenerates expert data with the
upstream collection settings (scripts/data/collect_tworooms.py:
ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05), 100-step episodes)
and compares the statistics. Agreement within sampling error means the
pinned numbers describe the data the model was trained on.
"""

import argparse
import warnings

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import yaml
from stable_worldmodel.envs.two_room import ExpertPolicy

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--protocol", default="protocols/tworoom/repo-k25.yaml")
parser.add_argument("--trajectories", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

pinned = yaml.safe_load(open(args.protocol))["normalization"]["action"]
env = gym.make("swm/TwoRoom-v1").unwrapped
rng = np.random.default_rng(args.seed)
policy = ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05, seed=args.seed)
policy.set_env(env)

actions = []
for _ in range(args.trajectories):
    policy._last_action = None
    _, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    for _ in range(100):
        a = policy.get_action({"state": info["state"], "goal_state": info["goal_state"]})
        actions.append(np.asarray(a, dtype=np.float64))
        _, _, terminated, _, info = env.step(a)
        if terminated:
            break

x = np.stack(actions)
mean, std = x.mean(axis=0), x.std(axis=0)
# Rough standard errors (rows within an episode are correlated; treat as iid).
se_mean = std / np.sqrt(len(x))
print(f"{len(x)} expert actions from {args.trajectories} trajectories")
for i in range(x.shape[1]):
    dm = mean[i] - pinned["mean"][i]
    ds = std[i] - pinned["std"][i]
    print(
        f"dim {i}: mean {mean[i]:+.4f} (pinned {pinned['mean'][i]:+.4f}, diff {dm:+.4f}, "
        f"~{abs(dm) / se_mean[i]:.1f} SE) | std {std[i]:.4f} (pinned {pinned['std'][i]:.4f}, "
        f"diff {ds:+.4f})"
    )
