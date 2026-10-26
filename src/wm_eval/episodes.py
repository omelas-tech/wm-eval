"""Episode packs: the exact start/goal pairs a protocol evaluates on.

Published TwoRoom numbers sample start/goal pairs from a multi-gigabyte
expert dataset, so reproducing them means downloading it. wm-eval instead
generates the pairs from the environment itself, deterministically from the
protocol card, and stores them as a few-kilobyte JSON "pack". The pack is the
source of truth at evaluation time: anyone can rerun the identical episodes,
and regenerating it on another machine is checked against its hash.

Two goal constructions are supported:

``expert_offset``
    Mirrors the released LeWM evaluation. Roll out the noisy TwoRoom expert
    that generated the training data, pick a start row uniformly among rows
    that have ``goal_offset`` steps of future, and use the agent position
    ``goal_offset`` steps later as the goal.

``env_native``
    The environment's own task: whatever start and target ``reset(seed)``
    samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .protocol import canonical_json, protocol_hash, sha256_text

PACK_SCHEMA = "wm-eval/pack@0"
_DECIMALS = 3  # positions are pixels; millipixel rounding keeps packs portable


def _pos(x) -> list[float]:
    return [round(float(v), _DECIMALS) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


def _make_env(card: dict):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/* envs)

    return gym.make(card["env"]["id"]).unwrapped


def _expert_trajectory(env, policy, seed: int, max_steps: int) -> list[list[float]]:
    _, info = env.reset(seed=seed)
    positions = [_pos(info["state"])]
    for _ in range(max_steps):
        action = policy.get_action({"state": info["state"], "goal_state": info["goal_state"]})
        _, _, terminated, truncated, info = env.step(action)
        positions.append(_pos(info["state"]))
        if terminated or truncated:
            break
    return positions


def _expert_offset(card: dict, env) -> list[dict]:
    from stable_worldmodel.envs.two_room import ExpertPolicy

    spec = card["episodes"]
    k = int(spec["goal_offset"])
    expert_cfg = spec["expert"]
    rng = np.random.default_rng(spec["seed"])
    policy = ExpertPolicy(
        action_noise=float(expert_cfg["action_noise"]),
        action_repeat_prob=float(expert_cfg["action_repeat_prob"]),
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    policy.set_env(env)

    trajectories = []
    for _ in range(int(spec["pool_trajectories"])):
        policy._last_action = None  # action-repeat memory must not leak across trajectories
        traj_seed = int(rng.integers(0, 2**31 - 1))
        trajectories.append(
            _expert_trajectory(env, policy, traj_seed, int(expert_cfg["max_steps"]))
        )

    # Uniform over valid (trajectory, row) pairs, like sampling dataset rows.
    valid = [(j, t) for j, traj in enumerate(trajectories) for t in range(len(traj) - k)]
    if len(valid) < spec["count"]:
        raise ValueError(
            f"only {len(valid)} valid start rows for {spec['count']} episodes; "
            "raise episodes.pool_trajectories"
        )
    picks = rng.choice(len(valid), size=spec["count"], replace=False)
    episodes = []
    for pick in sorted(int(p) for p in picks):
        j, t = valid[pick]
        episodes.append(
            {
                "start": trajectories[j][t],
                "goal": trajectories[j][t + k],
                "reset_seed": int(rng.integers(0, 2**31 - 1)),
                "source": {"trajectory": j, "row": t},
            }
        )
    return episodes


def _env_native(card: dict, env) -> list[dict]:
    spec = card["episodes"]
    rng = np.random.default_rng(spec["seed"])
    episodes = []
    for _ in range(spec["count"]):
        seed = int(rng.integers(0, 2**31 - 1))
        _, info = env.reset(seed=seed)
        episodes.append(
            {"start": _pos(info["state"]), "goal": _pos(info["goal_state"]), "reset_seed": seed}
        )
    return episodes


def pack_hash(episodes: list[dict]) -> str:
    core = [
        {"start": e["start"], "goal": e["goal"], "reset_seed": e["reset_seed"]} for e in episodes
    ]
    return sha256_text(canonical_json(core))


def build_pack(card: dict) -> dict:
    env = _make_env(card)
    construction = card["episodes"]["construction"]
    builder = {"expert_offset": _expert_offset, "env_native": _env_native}[construction]
    episodes = builder(card, env)
    env.close()
    return {
        "schema": PACK_SCHEMA,
        "protocol_id": card["id"],
        "protocol_hash": protocol_hash(card),
        "pack_hash": pack_hash(episodes),
        "episodes": episodes,
    }


def default_pack_path(card: dict, root: Path) -> Path:
    return root / f"{card['id'].replace('/', '__')}.{protocol_hash(card)[:12]}.json"


def save_pack(pack: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pack, indent=1) + "\n")


def load_pack(path: Path, card: dict) -> dict:
    pack = json.loads(Path(path).read_text())
    if pack.get("schema") != PACK_SCHEMA:
        raise ValueError(f"{path}: not a {PACK_SCHEMA} file")
    if pack["protocol_hash"] != protocol_hash(card):
        raise ValueError(f"{path}: built for protocol {pack['protocol_hash'][:12]}, not this card")
    if pack_hash(pack["episodes"]) != pack["pack_hash"]:
        raise ValueError(f"{path}: episodes were edited (pack hash mismatch)")
    if len(pack["episodes"]) != card["episodes"]["count"]:
        raise ValueError(f"{path}: episode count does not match the card")
    return pack
