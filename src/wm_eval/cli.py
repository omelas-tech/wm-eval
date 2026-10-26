"""wmeval command line.

wmeval hash    PROTOCOL                 print the protocol hash
wmeval pack    PROTOCOL [--out DIR]     build (or verify) the episode pack
wmeval run     PROTOCOL --policy REF    roll out a policy, write a card
wmeval report  [RESULTS] [--out FILE]   summarize cards as a markdown table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .protocol import load_protocol, protocol_hash

DEFAULT_CACHE = Path(os.environ.get("WM_EVAL_CACHE", Path.home() / ".cache" / "wm-eval"))


def _cmd_hash(args) -> int:
    print(protocol_hash(load_protocol(args.protocol)))
    return 0


def _get_pack(card: dict, packs: Path, rebuild: bool = False) -> tuple[dict, Path]:
    from .episodes import build_pack, default_pack_path, load_pack, save_pack

    path = default_pack_path(card, packs)
    if path.exists() and not rebuild:
        return load_pack(path, card), path
    pack = build_pack(card)
    save_pack(pack, path)
    return pack, path


def _cmd_pack(args) -> int:
    from .episodes import build_pack, default_pack_path, load_pack

    card = load_protocol(args.protocol)
    path = default_pack_path(card, Path(args.out))
    if path.exists() and not args.rebuild:
        stored = load_pack(path, card)
        if args.verify:
            fresh = build_pack(card)
            same = fresh["pack_hash"] == stored["pack_hash"]
            print(f"{path}: regenerated pack {'matches' if same else 'DIFFERS'}")
            return 0 if same else 1
        print(f"{path}: ok ({len(stored['episodes'])} episodes, pack {stored['pack_hash'][:12]})")
        return 0
    pack, path = _get_pack(card, Path(args.out), rebuild=True)
    print(f"{path}: built ({len(pack['episodes'])} episodes, pack {pack['pack_hash'][:12]})")
    return 0


def _cmd_run(args) -> int:
    from .card import make_card, save_card
    from .runner import run

    card = load_protocol(args.protocol)
    pack, pack_path = _get_pack(card, Path(args.packs))
    print(f"protocol {card['id']} {protocol_hash(card)[:12]} | pack {pack['pack_hash'][:12]}")

    def progress(done, total):
        print(f"  {args.policy}: {done}/{total} episodes", file=sys.stderr, flush=True)

    result = run(
        card,
        pack,
        args.policy,
        device=args.device,
        cache_dir=DEFAULT_CACHE / "checkpoints",
        progress=progress,
    )
    out = make_card(card, pack, result)
    path = save_card(out, Path(args.out))
    r = out["result"]
    lo, hi = r["ci95"]
    print(
        f"{args.policy}: {r['successes']}/{r['episodes']} = {100 * r['success_rate']:.1f}% "
        f"[95% CI {100 * lo:.1f}–{100 * hi:.1f}] in {result.seconds}s -> {path}"
    )
    return 0


def _cmd_report(args) -> int:
    from .card import load_cards, render_markdown, summarize

    rows = summarize(load_cards(Path(args.results)))
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    table = render_markdown(rows)
    if args.out:
        Path(args.out).write_text(table)
    print(table, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wmeval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hash", help="print a protocol card's hash")
    p.add_argument("protocol")
    p.set_defaults(func=_cmd_hash)

    p = sub.add_parser("pack", help="build or verify a protocol's episode pack")
    p.add_argument("protocol")
    p.add_argument("--out", default="packs")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--verify", action="store_true", help="regenerate and compare hashes")
    p.set_defaults(func=_cmd_pack)

    p = sub.add_parser("run", help="evaluate a policy and write a reproduction card")
    p.add_argument("protocol")
    p.add_argument("--policy", required=True, help="random | expert | hf:<user>/<repo> | folder")
    p.add_argument("--device", default="cpu")
    p.add_argument("--packs", default="packs")
    p.add_argument("--out", default="results")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("report", help="summarize reproduction cards")
    p.add_argument("results", nargs="?", default="results")
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
