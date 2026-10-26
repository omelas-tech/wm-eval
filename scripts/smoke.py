"""Smoke test: load the released TwoRoom LeWM (weights-only) and reset the env."""

import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

t = time.time()
import stable_worldmodel as swm  # noqa: E402

from wm_eval.models import load_world_model  # noqa: E402

print("import swm", round(time.time() - t, 1), "s")

lm = load_world_model("hf:quentinll/lewm-tworooms", Path.home() / ".cache/wm-eval/checkpoints")
n_params = sum(p.numel() for p in lm.model.parameters()) / 1e6
print("loaded", type(lm.model).__name__, "sha", lm.weights_sha256[:12], f"{n_params:.1f}M params")

world = swm.World("swm/TwoRoom-v1", num_envs=4, image_shape=(224, 224), max_episode_steps=100)
world.reset(seed=[0, 1, 2, 3])
print({k: getattr(v, "shape", None) for k, v in world.infos.items() if hasattr(v, "shape")})
print("proprio", world.infos["proprio"].reshape(4, -1))
