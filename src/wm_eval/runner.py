"""Roll out one policy on one episode pack under one protocol card.

Policies:

``random``  uniform actions; the floor every score is normalized against.
``expert``  the TwoRoom scripted expert steering to the true goal position
            (no noise); an oracle ceiling for what the pack allows.
``hf:<user>/<repo>`` or a checkpoint folder
            a latent world model planned with the card's solver/objective.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

BASELINES = ("random", "expert")
IMAGENET = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


@dataclass
class EpisodeResult:
    success: bool
    steps_to_success: int | None
    final_distance: float


@dataclass
class RunResult:
    policy: str
    episodes: list[EpisodeResult]
    seconds: float
    device: str
    model: dict = field(default_factory=dict)


def _img_transform(size: int):
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**IMAGENET),
            transforms.Resize(size=size),
        ]
    )


def _world_model_policy(card: dict, model, device: str, seed: int):
    import stable_worldmodel as swm
    from stable_worldmodel.data.normalization import ZScoreScaler
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    from stable_worldmodel.planning.solver import CEMSolver

    p = card["planner"]
    norm = card["normalization"]["action"]
    solver = CEMSolver(
        cost=ShootingCostEvaluator(model, GoalMSE()),
        batch_size=p["batch_size"],
        num_samples=p["num_samples"],
        var_scale=float(p["var_scale"]),
        n_steps=p["n_steps"],
        topk=p["topk"],
        device=device,
        seed=seed,
    )
    config = swm.PlanConfig(
        horizon=p["horizon"],
        receding_horizon=p["receding_horizon"],
        action_block=p["action_block"],
        history_len=p["history_len"],
        warm_start=p["warm_start"],
    )
    size = card["env"]["image_size"]
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process={"action": ZScoreScaler(mean=[norm["mean"]], std=[norm["std"]])},
        transform={"pixels": _img_transform(size), "goal": _img_transform(size)},
    )


def _baseline_policy(name: str, seed: int):
    import stable_worldmodel as swm
    from stable_worldmodel.envs.two_room import ExpertPolicy

    if name == "random":
        return swm.policy.RandomPolicy(seed=seed)
    return ExpertPolicy(action_noise=0.0, seed=seed)


def _goal_frames(world, goals: list[list[float]]) -> np.ndarray:
    """Render each env's goal image: the scene with the agent at the goal."""
    import torch

    frames = []
    for env, goal in zip(world.envs.envs, goals, strict=True):
        base = env.unwrapped
        img = base._render_frame(agent_pos=torch.tensor(goal, dtype=torch.float32))
        frames.append(np.asarray(img.cpu().numpy()).transpose(1, 2, 0))
    return np.stack(frames)[:, None]  # (n_envs, 1, H, W, C), like infos['pixels']


def _run_batch(card: dict, policy, episodes: list[dict]) -> list[EpisodeResult]:
    import stable_worldmodel as swm

    budget = card["eval_budget"]
    size = card["env"]["image_size"]
    world = swm.World(
        card["env"]["id"],
        num_envs=len(episodes),
        image_shape=(size, size),
        max_episode_steps=2 * budget,  # the budget, not a TimeLimit, ends episodes
    )
    world.set_policy(policy)
    world.reset(
        seed=[e["reset_seed"] for e in episodes],
        options=[{"state": e["start"], "target_state": e["goal"]} for e in episodes],
    )
    goal_pixels = _goal_frames(world, [e["goal"] for e in episodes])
    world.infos["goal"] = goal_pixels.copy()

    n = len(episodes)
    success = np.zeros(n, dtype=bool)
    first_hit = np.full(n, -1)
    distance = np.asarray(world.infos["distance_to_target"], dtype=float).reshape(n)
    step = {"t": 0}

    def on_step(w, mask):
        step["t"] += 1
        w.infos["goal"] = goal_pixels.copy()
        hit = np.asarray(w.terminateds, dtype=bool).reshape(n)
        newly = hit & ~success
        first_hit[newly] = step["t"]
        success[:] |= hit
        live = ~success
        distance[live] = np.asarray(w.infos["distance_to_target"], dtype=float).reshape(n)[live]
        distance[newly] = np.asarray(w.infos["distance_to_target"], dtype=float).reshape(n)[newly]

    world._run(max_steps=budget, mode="wait", on_step=on_step)
    world.close()
    return [
        EpisodeResult(
            success=bool(success[i]),
            steps_to_success=int(first_hit[i]) if first_hit[i] >= 0 else None,
            final_distance=round(float(distance[i]), 3),
        )
        for i in range(n)
    ]


def run(
    card: dict,
    pack: dict,
    policy_ref: str,
    device: str = "cpu",
    cache_dir=None,
    progress=None,
) -> RunResult:
    import torch

    from .models import load_world_model

    episodes = pack["episodes"]
    batch = card["run"]["batch_size"]
    model_info: dict = {}
    loaded = None
    if policy_ref not in BASELINES:
        loaded = load_world_model(policy_ref, cache_dir=cache_dir, device=device)
        model_info = {"ref": loaded.ref, "weights_sha256": loaded.weights_sha256}

    results: list[EpisodeResult] = []
    start = time.time()
    for b, lo in enumerate(range(0, len(episodes), batch)):
        # A fresh policy per batch: no warm-start or RNG state leaks between
        # batches, so a batch's outcome depends only on (card, pack, batch index).
        seed = card["planner"]["seed"] + b
        if loaded is None:
            policy = _baseline_policy(policy_ref, seed)
        else:
            policy = _world_model_policy(card, loaded.model, device, seed)
        with torch.inference_mode():
            results.extend(_run_batch(card, policy, episodes[lo : lo + batch]))
        if progress:
            progress(len(results), len(episodes))
    return RunResult(
        policy=policy_ref,
        episodes=results,
        seconds=round(time.time() - start, 1),
        device=device,
        model=model_info,
    )
