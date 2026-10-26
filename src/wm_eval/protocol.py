"""Protocol cards: every choice that can move a planning success rate, pinned.

A protocol card is a small YAML file. Its identity is the SHA-256 of its
canonical JSON form (sorted keys, no whitespace), ignoring only the
human-readable ``description`` and ``notes`` fields. Two runs that report the
same protocol hash were evaluated under the same rules; anything else is a
different protocol, whatever its name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "wm-eval/protocol@0"
HASH_EXCLUDE = frozenset({"description", "notes"})

GOAL_CONSTRUCTIONS = ("expert_offset", "env_native")
SOLVERS = ("cem",)
OBJECTIVES = ("goal_mse",)

# (key path, expected type) pairs every card must carry.
_REQUIRED: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("schema", str),
    ("id", str),
    ("env.id", str),
    ("env.image_size", int),
    ("episodes.count", int),
    ("episodes.seed", int),
    ("episodes.construction", str),
    ("eval_budget", int),
    ("normalization.action.mean", list),
    ("normalization.action.std", list),
    ("planner.solver", str),
    ("planner.objective", str),
    ("planner.num_samples", int),
    ("planner.n_steps", int),
    ("planner.topk", int),
    ("planner.var_scale", (int, float)),
    ("planner.horizon", int),
    ("planner.receding_horizon", int),
    ("planner.action_block", int),
    ("planner.history_len", int),
    ("planner.warm_start", bool),
    ("planner.batch_size", int),
    ("planner.seed", int),
    ("run.batch_size", int),
)


class ProtocolError(ValueError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _get(card: dict, path: str) -> Any:
    node: Any = card
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ProtocolError(f"missing field: {path}")
        node = node[part]
    return node


def validate(card: dict) -> dict:
    for path, typ in _REQUIRED:
        value = _get(card, path)
        # bool is an int subclass; never accept it where a number is expected.
        if typ is not bool and isinstance(value, bool):
            raise ProtocolError(f"{path}: expected {typ}, got bool")
        if not isinstance(value, typ):
            raise ProtocolError(f"{path}: expected {typ}, got {type(value).__name__}")
    if card["schema"] != SCHEMA:
        raise ProtocolError(f"unsupported schema {card['schema']!r} (want {SCHEMA})")

    construction = card["episodes"]["construction"]
    if construction not in GOAL_CONSTRUCTIONS:
        raise ProtocolError(f"episodes.construction must be one of {GOAL_CONSTRUCTIONS}")
    if construction == "expert_offset":
        for path in (
            "episodes.goal_offset",
            "episodes.pool_trajectories",
            "episodes.expert.action_noise",
            "episodes.expert.action_repeat_prob",
            "episodes.expert.max_steps",
        ):
            _get(card, path)

    planner = card["planner"]
    if planner["solver"] not in SOLVERS:
        raise ProtocolError(f"planner.solver must be one of {SOLVERS}")
    if planner["objective"] not in OBJECTIVES:
        raise ProtocolError(f"planner.objective must be one of {OBJECTIVES}")
    if planner["receding_horizon"] > planner["horizon"]:
        raise ProtocolError("planner.receding_horizon cannot exceed planner.horizon")
    if planner["horizon"] * planner["action_block"] > card["eval_budget"]:
        raise ProtocolError("planning horizon (horizon * action_block) exceeds eval_budget")
    if planner["topk"] > planner["num_samples"]:
        raise ProtocolError("planner.topk cannot exceed planner.num_samples")

    norm = card["normalization"]["action"]
    if len(norm["mean"]) != len(norm["std"]):
        raise ProtocolError("normalization.action mean/std length mismatch")
    if any(s <= 0 for s in norm["std"]):
        raise ProtocolError("normalization.action.std must be positive")
    return card


def protocol_hash(card: dict) -> str:
    return sha256_text(canonical_json({k: v for k, v in card.items() if k not in HASH_EXCLUDE}))


def load_protocol(path: str | Path) -> dict:
    card = yaml.safe_load(Path(path).read_text())
    if not isinstance(card, dict):
        raise ProtocolError(f"{path}: not a mapping")
    return validate(card)
