# TetraSwarm — Unknown-Maze Warehouse Mission (continuation handoff)

Read this + the auto-memory (`MEMORY.md` → `tetraswarm-project.md`) to continue.
Working dir: `/Users/kyawlinnkhant/Downloads/TetraSwarm`. Python: system `python3`
(has mujoco). For ViPER only: `/opt/anaconda3/envs/imp_mjc_rl/bin/python`.

> ## ✅ STATUS UPDATE (2026-06-13): MISSION BUILT + PASSING
> The two pieces below are now DONE:
> - **`dock_blocks` implemented** in `build_pointmaze_scene` (5 Tetris bodies packed
>   at the dock, each on its narrow side, 0 wall-overlap at spawn; `TETRIS_COLORS`
>   const; `_free_block_body` gained `yaw`; `info["blocks"]` populated).
> - **`scripts/demo_warehouse.py`** (NOT `demo_mission.py` — that name was already a
>   DIFFERENT working demo, the arena slalom; left untouched). Explore → destination
>   → one-by-one A* transport on the discovered grid → Tetris placement.
>   **All 5 blocks delivered 0.00 m, PASS.** `python3 scripts/demo_warehouse.py --gif`
>   → `results/mission.gif` + `mission_map.png`.
>
> Remaining: wire `demo_warehouse.py` into `render_all.sh` + README; delete unused
> frontier code in `demo_pointmaze.py`. The original plan (kept below) is now history.

---

## THE MISSION PLAN (what we're building)
An autonomous warehouse in an **unknown building**:
1. A fleet of **4 drones** starts at a **dock (bottom-left)** where **5 tetromino
   blocks (I, L, O, Z, T)** sit Tetris-packed, each a different Tetris colour.
2. The maze is **unknown** — the drones **explore + map** it first (they don't know
   where the far room is or how to get there).
3. The destination = the **furthest reachable cell (top-right)**, found from the
   *discovered* map.
4. The same fleet then **carries the 5 blocks one-by-one** from the dock to the
   destination, routing on the **discovered** map, and **places them side-by-side
   like Tetris** at the far end.

Decisions already made with the user:
- **4 drones** (8 was too crowded; 4 is faster + matches the relay swarm).
- **Wall-following** for exploration (frontier method capped ~30-40%, unreliable).
- **ViPER/Sartoretti angle is DROPPED** — "as long as it works in unknown env."
  (The real `external/ViPER` integration still exists and works, just not the focus.)
- Corridors are **wide (cell≈6.5 m)** so tetrominoes fit **without turn-to-fit**
  (I-bar with `block_tile≈0.4` is ~3.2 m < 6.5 m) — this SIMPLIFIES the transport.

---

## STATE: what's DONE vs TODO

### ✅ DONE — exploration (the prerequisite the user wanted fixed)
- **Farama PointMaze maze format** in `envs/scene_builder.py`:
  - `maze_from_grid(nx, ny, seed, n_goals, mission=False)` — DFS+braid maze as a
    PointMaze array (1=wall, 0=free, 'g'=goal, 'r'=start), full-cell wall blocks.
    `mission=True` → start (dock) at **bottom-left**, one goal **top-right**.
  - `build_pointmaze_scene(maze_map, cell=6.5, n_drones=4, ...)` → MuJoCo scene
    (wall blocks, goal sites hidden until discovered, lidar drones). Returns `info`
    with `nrows, ncols, cell, x0, y0, center(=start), goals, free_grid, angles, ...`.
  - `POINTMAZE_LARGE` constant (the canonical PointMaze LARGE map + g/r marks).
- **`scripts/demo_pointmaze.py`** — 4-drone **wall-following** explorer:
  half left-hand / half right-hand (`hand` array), `lidar_dir(d,phi)` reads range in
  a heading direction, thresholds `FRONT/TARGET/FAR` relative to `cell`, plus the
  reactive lidar-avoidance + separation. Discovers goal cells (`discover()` lights
  the `goal{k}` sites). True coverage = % of `info["free_grid"]` cells mapped-free.
  **RESULT: 98% mapped, 4/4 drop zones found, 0 collisions, ~68 s.**
  `python3 scripts/demo_pointmaze.py --gif` → `results/pointmaze.gif`,
  `results/pointmaze_map.png`.
  - NOTE: it still contains **unused frontier code** (`replan`, `gpath`, `gidx`,
    `find_frontiers`/`astar` imports) — safe to delete on cleanup.

### 🔧 IN PROGRESS — mission scene (half-done, finish this first)
- `build_pointmaze_scene` signature now has **`dock_blocks=None, block_tile=0.4`**
  params **but they are NOT implemented yet** (no block bodies built, not in `info`).
  **NEXT: implement them** — when `dock_blocks=["I","L","O","Z","T"]`, build 5
  tetromino free-bodies at the dock (`start` world pos), Tetris colours, and return
  `info["blocks"]` = {shape: (x,y)}. Reuse `_free_block_body(name, shape, pos,
  tile_half, half_z, color)`, `TETROMINOES`, `cell_offsets`, `grip_points`.
  Tetris colours: I cyan `0 0.8 0.9 1`, L orange `0.95 0.5 0.1 1`, O yellow
  `0.95 0.8 0.1 1`, Z red `0.9 0.15 0.2 1`, T purple `0.6 0.3 0.85 1`.
  Place them in a row at the dock (spacing ~1.3-1.8 m; small `block_tile` so they
  fit). They must NOT overlap the dock walls — the dock is the 'r' open cell.

### ⬜ TODO — the transport (the big remaining piece)
Build `scripts/demo_mission.py` (or extend `demo_pointmaze.py`) that does, in ONE run:
1. **Build** mission scene: `mz = maze_from_grid(nx=6, ny=5, seed=3, mission=True)`,
   `build_pointmaze_scene(mz, cell=6.5, n_drones=4, dock_blocks=["I","L","O","Z","T"])`.
2. **Explore** (the wall-following loop from `demo_pointmaze.py`) → build the
   occupancy `grid`. Stop when coverage high / destination region mapped.
3. **Destination** = the 'g' cell world pos (`info["goals"][0]`), OR compute the
   furthest reachable free cell from the dock on the DISCOVERED grid (BFS/A*).
4. **Transport one-by-one**: for each block (I,L,O,Z,T):
   - A* a route dock→destination on the **discovered** occupancy grid
     (`astar` from `demo_scout`, walls dilated; corridors are wide so blocks fit
     straight — NO turn-to-fit needed).
   - Drones fly to the block, **Hungarian** nearest-grip pickup (reuse
     `order_grips`/`morph_to` logic from `demo_maze_relay.py`), carry it
     **kinematically** along the route (drones + block follow the path, like the
     relay's `step()`), set it down at a **Tetris slot** offset from the destination
     (block k goes to slot k so they sit side-by-side).
   - Return empty (compact square) for the next block.
5. **Render** scene + map side-by-side, save `results/mission.gif`.

The transport machinery to REUSE is all in `scripts/demo_maze_relay.py`:
`step()` (kinematic drones+block), `gworld`, `order_grips` (Hungarian),
`morph_to`, `transit`, `follow`, `hold_phase`. The relay carries on GROUND-TRUTH
routes (`info["routes"]`) — for the mission, swap those for **A\* routes on the
discovered grid**.

---

## KEY GOTCHAS (bit us this session)
- **PIL collapses identical GIF frames** with `optimize=True` (50 identical → 1).
  For held frames use a **per-frame `duration` list** (one frame, long duration).
  See `render.py render_morph` for the pattern.
- **Frontier exploration is unreliable** on these mazes (clusters, stalls ~30-40%
  true coverage). Use **wall-following**. Don't go back to frontier.
- **Wide corridors** (cell 6.5) needed for 4-8 drones + block transport without
  collisions; narrow (4.5) → crowding + 100k wall hits.
- Coverage must be measured over `info["free_grid"]` cells, NOT the padded grid
  (the padded metric over-reports ~2-3×).
- `np.ptp(arr)` not `arr.ptp()` (NumPy 2.0). `data.geom_xpos` not `model.geom_xpos`.
- Render drones' colour marker (not the tiny X2 mesh) with
  `MjvOption.geomgroup[2]=0; [3]=1` if you need them visible at maze zoom — BUT the
  user wants **original real-X2 size** for transport (bigger meshes overlap during
  carry).

## OTHER WORKING PIECES (not part of this mission but in the repo)
- `scripts/render.py morph --drones 12` — LLM-driven formation morph (circle→…→
  fibonacci, looped, 0.47 m guaranteed clearance via altitude deconfliction).
- `scripts/demo_maze_relay.py --sweep --render` — the braided-maze full mission
  (ViPER sweep + completion + Hungarian relay). Separate from the PointMaze mission.
- `external/ViPER` — real marmotlab ViPER runs at default settings
  (`PYTHONPATH=_shim /opt/anaconda3/envs/imp_mjc_rl/bin/python run_viper.py`).

## SUGGESTED FIRST STEPS NEXT SESSION
1. Implement `dock_blocks` in `build_pointmaze_scene` + verify the scene renders
   (dock with 5 colour blocks, drones, maze). `python3 -c "from envs.scene_builder
   import *; build a mission scene; render top-down"`.
2. Write `scripts/demo_mission.py`: explore (copy wall-following loop) → destination
   → one-by-one A\*-routed kinematic transport → Tetris placement → `results/mission.gif`.
3. Then README/paper update for the warehouse mission.
