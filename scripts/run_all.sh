#!/usr/bin/env bash
# Rebuild every pack and evaluate every policy under every TwoRoom protocol.
# Usage: scripts/run_all.sh [device]   (default: mps on Apple silicon, else cpu)
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-$(python3 -c 'import platform;print("mps" if platform.machine()=="arm64" and platform.system()=="Darwin" else "cpu")')}"
MODEL="hf:quentinll/lewm-tworooms"
export STABLEWM_HOME="${STABLEWM_HOME:-$HOME/.cache/wm-eval/swm}" PYTHONWARNINGS=ignore
WMEVAL=.venv/bin/wmeval
PROTOCOLS=(protocols/tworoom/repo-k25.yaml protocols/tworoom/repo-k25-hist3.yaml
           protocols/tworoom/k50-b100.yaml protocols/tworoom/native-b100.yaml)
for p in "${PROTOCOLS[@]}"; do
  $WMEVAL pack "$p"
  for baseline in random expert; do $WMEVAL run "$p" --policy "$baseline"; done
done
for p in "${PROTOCOLS[@]}"; do
  $WMEVAL run "$p" --policy "$MODEL" --device "$DEVICE"
done
$WMEVAL report results --out results/README.md
