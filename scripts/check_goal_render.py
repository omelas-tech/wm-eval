"""Check that a rendered goal image is pixel-identical to the observation the
env produces when the agent actually stands at that position."""

import warnings

import numpy as np
import stable_worldmodel as swm

from wm_eval.runner import _goal_frames

warnings.filterwarnings("ignore")

world = swm.World("swm/TwoRoom-v1", num_envs=2, image_shape=(224, 224), max_episode_steps=100)
goals = [[60.0, 60.0], [170.0, 150.0]]
world.reset(seed=[0, 1], options=[{"state": g, "target_state": g} for g in goals])
obs = world.infos["pixels"]
rendered = _goal_frames(world, goals)
print("obs", obs.shape, obs.dtype, obs.min(), obs.max())
print("goal", rendered.shape, rendered.dtype, rendered.min(), rendered.max())
diff = np.abs(obs.astype(np.int16) - rendered.astype(np.int16))
print("max abs pixel diff:", int(diff.max()), "| identical:", bool(diff.max() == 0))
