from dataclasses import dataclass, field
from pathlib import Path

import pytest

from wm_eval.card import load_cards, make_card, render_markdown, save_card, summarize
from wm_eval.episodes import load_pack, pack_hash
from wm_eval.protocol import load_protocol, protocol_hash

ROOT = Path(__file__).parents[1]


@dataclass
class Ep:
    success: bool
    steps_to_success: int | None = None
    final_distance: float = 0.0


@dataclass
class Result:
    policy: str
    episodes: list
    seconds: float = 1.0
    device: str = "cpu"
    model: dict = field(default_factory=dict)


@pytest.fixture
def card():
    return load_protocol(ROOT / "protocols/tworoom/repo-k25.yaml")


def _pack(card, n):
    eps = [{"start": [1.0, 2.0], "goal": [3.0, float(i)], "reset_seed": i} for i in range(n)]
    return {"pack_hash": pack_hash(eps), "episodes": eps, "protocol_hash": protocol_hash(card)}


def test_card_roundtrip_and_summary(tmp_path, card):
    pack = _pack(card, 10)
    runs = {
        "random": [True] * 2 + [False] * 8,
        "expert": [True] * 10,
        "hf:me/model": [True] * 8 + [False] * 2,
    }
    for ref, outcomes in runs.items():
        res = Result(policy=ref, episodes=[Ep(o) for o in outcomes])
        save_card(make_card(card, pack, res), tmp_path)

    rows = summarize(load_cards(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "hf:me/model"
    assert row["success"].startswith("80%")
    assert row["normalized"] == pytest.approx(0.75)  # (0.8 - 0.2) / (1.0 - 0.2)
    assert row["vs_random_p"] == pytest.approx(2 * 0.5**6)  # 6 discordant, all for model
    table = render_markdown(rows)
    assert "tworoom/repo-k25" in table and "0.75" in table


def test_runs_on_different_packs_are_not_mixed(tmp_path, card):
    for n, ref in ((10, "random"), (12, "hf:me/model")):
        res = Result(policy=ref, episodes=[Ep(True)] * n)
        save_card(make_card(card, _pack(card, n), res), tmp_path)
    (row,) = summarize(load_cards(tmp_path))
    assert row["random"] == "—" and row["normalized"] is None


def test_shipped_packs_verify():
    for path in sorted((ROOT / "packs").glob("*.json")):
        stem = path.name.split(".")[0].replace("__", "/")
        proto = load_protocol(ROOT / "protocols" / f"{stem}.yaml")
        load_pack(path, proto)  # raises on any hash or count mismatch


def test_tampered_pack_is_rejected(tmp_path, card):
    import json

    path = next((ROOT / "packs").glob("tworoom__repo-k25.*.json"), None)
    if path is None:
        pytest.skip("pack not built")
    pack = json.loads(path.read_text())
    pack["episodes"][0]["goal"][0] += 1.0
    bad = tmp_path / path.name
    bad.write_text(json.dumps(pack))
    with pytest.raises(ValueError, match="edited"):
        load_pack(bad, card)
