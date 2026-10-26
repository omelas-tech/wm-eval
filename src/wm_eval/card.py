"""Reproduction cards: one JSON file per (protocol, pack, policy) run.

A card records what was evaluated (protocol hash, pack hash, weights hash,
software versions) next to what happened (per-episode outcomes), so a number
can always be traced back to the exact conditions that produced it.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path

from .protocol import protocol_hash
from .stats import mcnemar_exact, normalized, wilson

CARD_SCHEMA = "wm-eval/card@0"


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version(), "platform": platform.platform(terse=True)}
    for pkg in ("wm-eval", "stable-worldmodel", "torch", "transformers", "gymnasium"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def policy_slug(policy_ref: str) -> str:
    return policy_ref.replace("hf:", "").replace("/", "__").strip("_.") or "policy"


def make_card(card: dict, pack: dict, result) -> dict:
    outcomes = [e.success for e in result.episodes]
    k, n = sum(outcomes), len(outcomes)
    lo, hi = wilson(k, n)
    return {
        "schema": CARD_SCHEMA,
        "protocol": {"id": card["id"], "hash": protocol_hash(card)},
        "pack": {"hash": pack["pack_hash"], "episodes": n},
        "policy": {"ref": result.policy, **result.model},
        "result": {
            "successes": k,
            "episodes": n,
            "success_rate": k / n,
            "ci95": [lo, hi],
        },
        "runtime": {"device": result.device, "seconds": result.seconds, **_versions()},
        "episodes": [asdict(e) for e in result.episodes],
    }


def save_card(card_json: dict, root: Path) -> Path:
    proto = card_json["protocol"]
    path = (
        root / proto["id"] / f"{policy_slug(card_json['policy']['ref'])}.{proto['hash'][:12]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card_json, indent=1) + "\n")
    return path


def load_cards(root: Path) -> list[dict]:
    cards = []
    for path in sorted(Path(root).rglob("*.json")):
        data = json.loads(path.read_text())
        if data.get("schema") == CARD_SCHEMA:
            cards.append(data)
    return cards


def _fmt_rate(res: dict) -> str:
    lo, hi = res["ci95"]
    return f"{100 * res['success_rate']:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}]"


def summarize(cards: list[dict]) -> list[dict]:
    """One row per (protocol, pack, model) with the baselines from the same pack."""
    by_pack: dict[tuple[str, str, str], dict[str, dict]] = {}
    for c in cards:
        key = (c["protocol"]["id"], c["protocol"]["hash"], c["pack"]["hash"])
        by_pack.setdefault(key, {})[c["policy"]["ref"]] = c

    rows = []
    for (pid, phash, pkhash), runs in sorted(by_pack.items()):
        floor = runs.get("random")
        ceiling = runs.get("expert")
        for ref, c in sorted(runs.items()):
            if ref in ("random", "expert"):
                continue
            res = c["result"]
            row = {
                "protocol": pid,
                "protocol_hash": phash[:12],
                "pack_hash": pkhash[:12],
                "model": ref,
                "n": res["episodes"],
                "success": _fmt_rate(res),
                "random": _fmt_rate(floor["result"]) if floor else "—",
                "expert": _fmt_rate(ceiling["result"]) if ceiling else "—",
                "normalized": None,
                "vs_random_p": None,
            }
            if floor:
                ceil_rate = ceiling["result"]["success_rate"] if ceiling else 1.0
                norm = normalized(res["success_rate"], floor["result"]["success_rate"], ceil_rate)
                row["normalized"] = None if norm is None else round(norm, 2)
                test = mcnemar_exact(
                    [e["success"] for e in c["episodes"]],
                    [e["success"] for e in floor["episodes"]],
                )
                row["vs_random_p"] = test["p_value"]
            rows.append(row)
    return rows


def paired_comparisons(cards: list[dict]) -> list[dict]:
    """Same model, same episodes, different protocol: exact McNemar per pair.

    Protocols that differ only in planner settings share an episode pack, so
    their outcomes pair episode by episode and the comparison isolates the
    planner setting from episode sampling noise.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for c in cards:
        if c["policy"]["ref"] in ("random", "expert"):
            continue
        groups.setdefault((c["policy"]["ref"], c["pack"]["hash"]), []).append(c)

    out = []
    for (ref, pkhash), group in sorted(groups.items()):
        group = sorted(group, key=lambda c: c["protocol"]["id"])
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                test = mcnemar_exact(
                    [e["success"] for e in a["episodes"]],
                    [e["success"] for e in b["episodes"]],
                )
                out.append(
                    {
                        "model": ref,
                        "pack_hash": pkhash[:12],
                        "a": a["protocol"]["id"],
                        "b": b["protocol"]["id"],
                        "rate_a": a["result"]["success_rate"],
                        "rate_b": b["result"]["success_rate"],
                        **test,
                    }
                )
    return out


def render_markdown(rows: list[dict], pairs: list[dict] | None = None) -> str:
    table = _render_rows(rows)
    if not pairs:
        return table
    lines = [
        "",
        "Paired comparisons (same model, same episodes, different protocol; exact McNemar):",
        "",
        "| model | protocol A | protocol B | A | B | only A | only B | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in pairs:
        lines.append(
            f"| `{p['model']}` | `{p['a']}` | `{p['b']}` | {100 * p['rate_a']:.0f}% "
            f"| {100 * p['rate_b']:.0f}% | {p['only_a']} | {p['only_b']} | {p['p_value']:.2g} |"
        )
    return table + "\n".join(lines) + "\n"


def _render_rows(rows: list[dict]) -> str:
    head = (
        "| protocol | model | n | success [95% CI] | random | expert oracle "
        "| normalized | p vs random |\n"
        "|---|---|---:|---|---|---|---:|---:|\n"
    )
    lines = []
    for r in rows:
        p = "—" if r["vs_random_p"] is None else f"{r['vs_random_p']:.2g}"
        norm = "—" if r["normalized"] is None else f"{r['normalized']:.2f}"
        lines.append(
            f"| `{r['protocol']}` <sub>{r['protocol_hash']}</sub> | `{r['model']}` | {r['n']} "
            f"| {r['success']} | {r['random']} | {r['expert']} | {norm} | {p} |"
        )
    return head + "\n".join(lines) + "\n"
