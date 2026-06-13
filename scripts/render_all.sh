#!/usr/bin/env bash
# Wipe results/ and regenerate every TetraSwarm render + figure.
#   bash scripts/render_all.sh
# (no `set -e`: keep going even if one render fails, so the rest still complete)
cd "$(dirname "$0")/.."

echo "== wiping results/ =="
rm -rf results
mkdir -p results/figures

echo "== formation morph (circle -> square -> heart -> star, collision-free) =="
python3 scripts/render.py morph --drones 12

echo "== transport per tetromino shape =="
for s in I O T L S Z; do python3 scripts/render.py transport --shape "$s"; done

echo "== maze carry per tetromino shape =="
for s in I O T L S Z; do python3 scripts/render.py maze --shape "$s"; done

echo "== navigate / mission / squeeze / maze relay =="
python3 scripts/render.py navigate --drones 6
python3 scripts/render.py mission --shape Z
python3 scripts/render.py squeeze --gap 3.5
python3 scripts/demo_maze_relay.py --render

echo "== research figures + unknown-payload graphs =="
python3 scripts/figures.py
python3 scripts/demo_unknown.py --seed 3 --graphs

echo "== done. results/: =="
ls -1 results results/figures
