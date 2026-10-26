import copy
from pathlib import Path

import pytest

from wm_eval.protocol import ProtocolError, load_protocol, protocol_hash, validate

PROTOCOLS = sorted(Path(__file__).parents[1].glob("protocols/**/*.yaml"))


@pytest.fixture
def card():
    return load_protocol(Path(__file__).parents[1] / "protocols/tworoom/repo-k25.yaml")


@pytest.mark.parametrize("path", PROTOCOLS, ids=lambda p: p.stem)
def test_shipped_protocols_validate(path):
    load_protocol(path)


def test_shipped_protocol_hashes_are_distinct():
    hashes = [protocol_hash(load_protocol(p)) for p in PROTOCOLS]
    assert len(set(hashes)) == len(hashes)


def test_hash_ignores_prose_but_not_rules(card):
    base = protocol_hash(card)
    prose = copy.deepcopy(card)
    prose["description"] = "reworded"
    prose["notes"] = "reworded"
    assert protocol_hash(prose) == base

    rules = copy.deepcopy(card)
    rules["planner"]["history_len"] = 3
    assert protocol_hash(rules) != base


def test_hash_is_key_order_independent(card):
    reordered = dict(reversed(list(card.items())))
    assert protocol_hash(reordered) == protocol_hash(card)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["planner"].pop("seed"), "missing field: planner.seed"),
        (lambda c: c["planner"].update(horizon=20), "exceeds eval_budget"),
        (lambda c: c["planner"].update(receding_horizon=9), "cannot exceed planner.horizon"),
        (lambda c: c["planner"].update(topk=301), "topk"),
        (lambda c: c["planner"].update(warm_start=1), "expected"),
        (lambda c: c["planner"].update(num_samples=True), "got bool"),
        (lambda c: c["episodes"].update(construction="dataset"), "construction"),
        (lambda c: c["episodes"].pop("goal_offset"), "goal_offset"),
        (lambda c: c["normalization"]["action"].update(std=[1.0, 0.0]), "positive"),
        (lambda c: c.update(schema="other@1"), "unsupported schema"),
    ],
)
def test_invalid_cards_are_rejected(card, mutate, message):
    bad = copy.deepcopy(card)
    mutate(bad)
    with pytest.raises(ProtocolError, match=message):
        validate(bad)
