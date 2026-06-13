# Plan: use the REAL ViPER (marmotlab/ViPER), not a homemade explorer

## Why
- **ViPER = "Visibility-based Pursuit-Evasion via Reinforcement Learning"** (MarmotLab,
  Sartoretti). Learned RL policy (PyTorch) + **pretrained model** provided. Multi-agent
  team **sweeps/clears** an unknown env (contaminated→cleared, frontier expansion) so no
  worst-case evader can hide. Runs in ITS OWN grid sim on **PNG maps**;
  `test_driver.py` + `test_parameter.py` (defaults), `viper_demo.py` interactive.
- Our current `demo_scout.py` is a **classical frontier explorer + potential-field
  avoidance** — NOT ViPER, and a different task (mapping vs pursuit-evasion). The reactive
  avoidance is a hack ("too reactive"). Goal: replace with genuine ViPER at default settings.

## KEY DECISION (ask the user first)
ViPER's task is **pursuit-evasion clearing**, ours was framed as **mapping for transport**.
They overlap (clearing fully covers the map) but the framing differs. Decide:
- (a) Adopt ViPER's real framing: drones **sweep/clear** the unknown maze (guarantee no
  intruder), which also yields the map → then transport. Most honest to what ViPER is.
- (b) Use ViPER only for its coverage and keep an "exploration" framing (weaker / slightly
  misrepresents ViPER). Avoid.
Recommend (a).

## Integration mode (recommend A, then B if time)
- **A. ViPER standalone (cleanest, do first):** run marmotlab/ViPER in its own venv on a
  map exported from our maze, **default `test_parameter.py` + pretrained model**. Capture
  its native gif + agent paths + cleared-coverage map + metrics. This is the credible
  "ViPER" artifact.
- **B. MuJoCo replay:** import ViPER's agent trajectories and replay them in our MuJoCo
  maze (drones follow ViPER's grid-valid paths). **Delete the reactive-avoidance layer** —
  ViPER paths are collision-free on the grid; in MuJoCo just PD-track them with the wall
  inflation already baked into the grid. Unified visual.
- **C. Full pipeline (optional):** feed ViPER's resulting occupancy/cleared map into the
  courier relay = explore/clear-then-transport on ONE maze.

## Steps (next session)
1. **Env isolation:** clone `marmotlab/ViPER` (likely as `external/ViPER`, gitignored or
   submodule). Separate venv, Python 3.11, `pip install` their requirements (torch, ray,
   scikit-image, imageio, tensorboard, matplotlib, wandb, opencv-python-headless). Keep it
   OFF the mujoco env to avoid torch/ray conflicts; communicate via files.
2. **Reproduce default:** `./utils/download.sh` (pretrained model) → run `viper_demo.py`
   and `test_driver.py` on a sample map with **unchanged defaults**. Confirm it runs
   (user already saw "it's so fast"). Record the default config values from
   `test_parameter.py` (n_agents, sensor range, map resolution, action space) — these are
   the "default settings" we must keep.
3. **Maze → ViPER map:** add `scripts/export_maze_png.py` to render our braided maze
   (`build_scout_scene`/`build_maze_relay_scene` walls) as a binary PNG (free=white,
   obstacle=black) at ViPER's expected resolution; set start cells = depot centre.
4. **Run ViPER on our maze** at default settings → save native gif + trajectories
   (e.g. JSON of per-agent (x,y,t)) + final cleared map + metrics (steps-to-clear, % cleared).
5. **(B) MuJoCo replay:** `scripts/demo_viper_sweep.py` loads the trajectories, drives the
   solid scout drones along them in the MuJoCo maze. No potential-field avoidance.
6. **Retire/relabel** `demo_scout.py`: either delete, or keep clearly as a "classical
   frontier baseline" for an honest baseline-vs-learned-ViPER comparison (nice, skilled
   story — but only if framed as a baseline, never as ViPER).
7. **Docs:** README + paper — the sweep/clear stage uses **MarmotLab ViPER (pretrained,
   default settings)**; cite the ViPER paper; credit + respect license; precise wording
   (pursuit-evasion clearing, not generic exploration). Drop the "reactive avoidance" claims.

## Risks / to verify
- ViPER map format + resolution + free/obstacle convention (check their `maps/` dataset).
- ViPER default **agent count** (may not be 4 — respect their default or note the change).
- Trajectory export: ViPER may not emit paths by default → may need a small hook in
  `test_driver.py` to dump per-step agent positions.
- Pursuit-evasion vs mapping framing (the KEY DECISION above).
- License for using/redistributing their pretrained model (likely fine to USE; don't
  commit their weights — use `download.sh`).
- Env conflicts: keep ViPER venv separate; bridge via PNG in / JSON+gif out.

## What NOT to do
- Don't present the homemade frontier+reactive scout as "ViPER".
- Don't change ViPER's default hyperparameters (the whole point is "default settings").
- Don't commit ViPER weights or code into this repo's history (submodule / external dir).
