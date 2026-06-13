# Maze Navigation, Turn-to-Fit & the VIPER Scout

How TetraSwarm carries an oversized payload through a cluttered, **solid** (fully
collidable) environment — including rotating it to squeeze through gaps too narrow
for its wide side — and the plan for doing it in *unknown* mazes with a scout.

## Everything is reactive (solid) now

All bodies physically collide, so a bad plan actually goes wrong instead of
phasing through:

| Body | Collides? | Notes |
|------|-----------|-------|
| Drones (mass box) | ✅ | bump each other / world; spawned pre-spaced so they don't collide on fly-in |
| Payload (Z slab) | ✅ | rests on ground, blocked by walls it doesn't fit through |
| Maze / arena / door walls | ✅ (`solid=True`) | real obstacles |
| Drone visual mesh, tethers | ❌ | cosmetic only |

Stability comes from `integrator="implicitfast"`, soft contacts
(`solref="0.02 1"`), small timesteps, and a controller force clamp (`SwarmPD.fmax`)
so a collision *reacts* (bounces / goes haywire) rather than NaN-crashing. The
hard 1.5 m inter-drone spacing rule (`formations._enforce_min_sep`) keeps
formations collision-free at steady state.

## The Cleveland Z payload

A thin flat **slab** of 4 square tiles (`plan_transport`): tile edge 1.5 m,
5 cm thick, light foam (~12 kg/m³) → **≈5.4 kg, carried by 4 drones**, one over
each tile centre with a **vertical suction cable** (1.5 m drone spacing = safe
separation). The slab is **4.5 m wide (long axis) × 3 m (short axis)**; its
diagonal is ~5.4 m (matters for rotating in tight spaces).

## Turn-to-fit

A doorway narrower than the slab is wide forces a 90° rotation so the **narrow
3 m side leads**. Implemented as a yaw on the grip formation
(`make_transport_stepper(..., yaw_of=...)`): rotating the 4 grip points rotates
the rigid slab, so **turning the Z turns the drones with it**. Key rules:

- **Finish the turn early** — mid-rotation the slab's *diagonal* (5.4 m) is wider
  than either side, so the turn must complete a slab-length before the gap.
- **Slow down while turning** — carry progress eases to 30% speed during a turn
  (gentle, low tilt).
- A space to rotate in must be **larger than the slab diagonal** (~5.4 m), which
  is why maze cells are 7 m.

`scripts/demo_squeeze.py`: a single solid doorway. At a 3.5 m gap the Z turns and
fits; at 2.5 m (< its 3 m narrow side) it's **physically blocked** — proof the
collision is real, not scripted.

## The braided maze

`build_maze_scene` procedurally generates a **braided maze** — a perfect maze
(randomized DFS) with every dead-end opened up, so there are **no dead-ends**,
loops exist, and the scout always has somewhere productive to explore. Doorway
widths vary (some > the slab, some narrow enough to require a turn). A BFS finds
the route start→goal.

`scripts/demo_maze.py`: the couriers carry the Z along the route keeping its
**narrow side facing the direction of travel** (`yaw = heading`). When the route
turns a corner, the swarm rotates the slab 90° to thread the next doorway. Maze
walls are solid, so it genuinely has to fit.

## The VIPER scout (planned — Checkpoint 2)

Today a red "scout" drone flies the route as a placeholder. The plan (approved):

- **VIPER = MarmotLab informative-path-planning**, *not* a drone name. The scout
  is the only agent that explores/maps.
- Give it a simulated **lidar** (MuJoCo `rangefinder`s) and a **VIPER-style
  frontier / information-gain** explorer to map the maze into an occupancy grid
  (with sensor noise — not an oracle).
- Extract the route + per-gap widths/orientations from *that map*.
- The couriers (already map-consuming via `yaw_of` + route) plan their turns from
  the scout's map → works in **unknown environments with no overhead camera**.

This keeps a clean split: **scout maps, couriers consume**. A learned VIPER RL
policy is a later swap for the heuristic explorer.

## Run it

```bash
python scripts/demo_maze.py                 # carry the Z through the braided maze
python scripts/demo_squeeze.py --gap 3.5    # turn 90° through a solid doorway
python scripts/demo_squeeze.py --gap 2.5    # too narrow -> blocked (real collision)
python scripts/demo_mission.py --shape Z    # slalom mission (scout maps first)
python scripts/demo_navigate.py --drones 6  # formation through solid gates

python scripts/render.py squeeze            # -> results/*.gif (offscreen)
# maze GIF: see the inline render in scripts (results/maze.gif)
```
