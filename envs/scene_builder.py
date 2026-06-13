"""
TetraSwarm — MuJoCo scene generator.

Builds an MJCF scene with N drones + one rigid Tetris (tetromino) payload + a
ground plane. Drones are simplified quadrotors (a box body with a free joint);
they are force-controlled via `data.xfrc_applied` by the controller, which keeps
the prototype simple while we focus on coordination / formation / navigation.
A nicer rotor-level model (Skydio X2 from MuJoCo Menagerie) can be swapped in
later for visuals without changing the pipeline.

Tetromino payloads are a single rigid body made of welded unit cubes, so the
asymmetric inertia that makes formation/orientation matter comes for free.
"""
import os
import numpy as np

# Real Skydio X2 drone model (MuJoCo Menagerie), vendored under assets/.
_X2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "assets", "menagerie_tmp", "skydio_x2", "assets")


def _x2_asset_block(scale=0.01):
    """Compiler + asset XML that pulls in the X2 mesh & texture (absolute paths,
    so the scene loads from an XML string regardless of CWD). ``scale`` enlarges
    the real X2 model for visibility in wide top-down scenes."""
    return f'''  <compiler meshdir="{_X2_DIR}" texturedir="{_X2_DIR}" autolimits="true"/>
  <asset>
    <texture type="2d" name="x2tex" file="X2_lowpoly_texture_SpinningProps_1024.png"/>
    <material name="x2mat" texture="x2tex"/>
    <mesh name="X2" file="X2_lowpoly.obj" scale="{scale} {scale} {scale}"/>
  </asset>'''


# Tetromino footprints as (col, row) unit-cube offsets.
TETROMINOES = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(0, 0), (1, 0), (2, 0), (1, 1)],
    "L": [(0, 0), (0, 1), (0, 2), (1, 0)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 1), (1, 1), (1, 0), (2, 0)],   # "Cleveland Z"
}

# Distinct colors so individual drones are easy to follow in the viewer.
_DRONE_COLORS = [
    "0.90 0.30 0.30 1", "0.30 0.55 0.90 1", "0.30 0.80 0.40 1",
    "0.95 0.75 0.20 1", "0.70 0.40 0.85 1", "0.30 0.80 0.80 1",
    "0.95 0.55 0.75 1", "0.60 0.60 0.65 1", "0.85 0.50 0.25 1",
    "0.55 0.75 0.30 1",
]

# Canonical Tetris colours (used at the warehouse dock + far-end placement).
TETRIS_COLORS = {
    "I": "0 0.8 0.9 1",      # cyan
    "L": "0.95 0.5 0.1 1",   # orange
    "O": "0.95 0.8 0.1 1",   # yellow
    "Z": "0.9 0.15 0.2 1",   # red
    "T": "0.6 0.3 0.85 1",   # purple
}

CUBE = 0.15          # half-extent of a payload unit cube (m)  -> 0.30 m cubes
DRONE_HALF = 0.10    # half-extent of a drone body (m)
DRONE_MASS = 0.30    # kg


def cell_offsets(shape, cube_half=CUBE):
    """Local (x, y) of each cube center, relative to the footprint centroid."""
    cells = TETROMINOES[shape]
    cx = np.mean([c for c, _ in cells])
    cy = np.mean([r for _, r in cells])
    return [((c - cx) * 2 * cube_half, (r - cy) * 2 * cube_half) for (c, r) in cells]


def grip_points(cells, n, cube_half):
    """Spread ``n`` grip points across the payload cells (round-robin over cells,
    sub-gridded within a cell when it carries more than one drone). Lets any
    number of drones share the load, not just one per cube."""
    n_cells = len(cells)
    base, extra = divmod(n, n_cells)
    per = [base + (1 if k < extra else 0) for k in range(n_cells)]
    span = 1.2 * cube_half                       # spread within a cube's top face
    grips = []
    for (cx, cy), k in zip(cells, per):
        if k <= 0:
            continue
        if k == 1:
            grips.append((cx, cy))
            continue
        m = int(np.ceil(np.sqrt(k)))
        step = span / (m - 1) if m > 1 else 0.0
        idx = 0
        for r in range(m):
            for c in range(m):
                if idx >= k:
                    break
                grips.append((cx + (c - (m - 1) / 2) * step,
                              cy + (r - (m - 1) / 2) * step))
                idx += 1
    return grips


def plan_transport(shape="Z", tile_edge=1.5, thickness=0.05, density=12.0,
                   lift_per_drone=1.8, margin=1.2):
    """Size the job: how heavy is the slab, and how many drones does it need?

    The payload is a thin flat slab made of one square *tile* per tetromino cell
    (a light foam panel, default ~12 kg/m^3). One drone hovers over each tile's
    center with a vertical suction cable, so the tile edge IS the drone-to-drone
    spacing (1.5 m default -> safe separation). Mass = edge^2 * thickness *
    density per tile. Drone count = max(one per tile, capacity requirement).
    """
    n_cells = len(TETROMINOES[shape])
    mass_per_tile = tile_edge ** 2 * thickness * density
    payload_mass = n_cells * mass_per_tile
    capacity_count = int(np.ceil(payload_mass * margin / lift_per_drone))
    n_drones = max(n_cells, capacity_count)
    return {
        "shape": shape,
        "cube_edge": tile_edge,          # tile edge == drone spacing
        "thickness": thickness,
        "density": density,
        "n_cells": n_cells,
        "mass_per_cube": mass_per_tile,
        "payload_mass": payload_mass,
        "lift_per_drone": lift_per_drone,
        "margin": margin,
        "n_drones": n_drones,
        "share_per_drone": payload_mass / n_drones,
    }


def _payload_body(shape="L", spawn=(0.0, 0.0, 0.4), cube_mass=None, cube_half=CUBE,
                  half_z=None):
    """One rigid body of welded tiles/cubes; centered on its footprint centroid.
    ``half_z`` sets the vertical half-extent (a thin slab when small); defaults to
    ``cube_half`` (a cube)."""
    hz = cube_half if half_z is None else half_z
    mass_attr = f' mass="{cube_mass}"' if cube_mass is not None else ""
    geoms = []
    for (x, y) in cell_offsets(shape, cube_half):
        geoms.append(
            f'      <geom type="box" size="{cube_half} {cube_half} {hz}" '
            f'pos="{x:.3f} {y:.3f} 0"{mass_attr} rgba="0.80 0.80 0.85 1"/>'
        )
    geoms_xml = "\n".join(geoms)
    sx, sy, sz = spawn
    return f'''    <body name="payload" pos="{sx} {sy} {sz}">
      <freejoint name="payload_free"/>
      <site name="payload_center" pos="0 0 0" size="0.05" rgba="1 0 0 1"/>
{geoms_xml}
    </body>'''


def _tether_geoms(vec):
    """A thin cable from the drone center to a suction cup at ``vec`` (the grip
    point in the drone's local frame; may be angled). Purely visual — the carry
    force goes through the connect constraint."""
    dx, dy, dz = vec
    return (
        f'      <geom type="cylinder" fromto="0 0 0 {dx:.3f} {dy:.3f} {dz:.3f}" '
        f'size="0.006" rgba="0.05 0.05 0.05 1" contype="0" conaffinity="0" mass="0"/>\n'
        f'      <geom type="ellipsoid" size="0.05 0.05 0.02" '
        f'pos="{dx:.3f} {dy:.3f} {dz:.3f}" rgba="0.15 0.15 0.18 1" '
        f'contype="0" conaffinity="0" mass="0"/>'
    )


def _drone_body(i, pos, model="x2", tether=None, collide=True):
    color = _DRONE_COLORS[i % len(_DRONE_COLORS)]
    x, y, z = pos
    tether = ("\n" + _tether_geoms(tether)) if tether else ""
    nocol = "" if collide else ' contype="0" conaffinity="0"'
    if model == "x2":
        # Real X2 mesh for looks (massless, non-colliding) + an invisible mass
        # box that carries the mass AND (optionally) collides — so drones react
        # when they hit each other or the world.
        body_geoms = (
            f'      <geom type="mesh" mesh="X2" material="x2mat" quat="0 0 1 1" '
            f'contype="0" conaffinity="0" mass="0" group="2"/>\n'
            f'      <geom type="box" size="0.13 0.13 0.04" mass="{DRONE_MASS}"{nocol} '
            f'rgba="{color[:-1]}0.25" group="3"/>'
        )
    else:
        body_geoms = (f'      <geom type="box" size="{DRONE_HALF} {DRONE_HALF} 0.03" '
                      f'mass="{DRONE_MASS}" rgba="{color}"/>')
    return f'''    <body name="drone{i}" pos="{x} {y} {z}">
      <freejoint name="drone{i}_free"/>
{body_geoms}{tether}
      <site name="drone{i}_center" pos="0 0 0" size="0.02"/>
    </body>'''


def build_scene(n_drones=10, payload_shape="L", with_payload=True, ring_radius=1.5,
                drone_model="x2"):
    """Return an MJCF XML string for a TetraSwarm scene."""
    # Spawn drones evenly on a ring so nothing overlaps at t=0.
    drones = []
    for i in range(n_drones):
        a = 2 * np.pi * i / n_drones
        drones.append(_drone_body(i, (ring_radius * np.cos(a), ring_radius * np.sin(a), 1.0),
                                  model=drone_model))
    drones_xml = "\n".join(drones)
    payload_xml = _payload_body(payload_shape) if with_payload else ""
    assets = _x2_asset_block() if drone_model == "x2" else ""

    return f'''<mujoco model="tetraswarm">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <default>
    <geom solref="0.02 1" solimp="0.8 0.9 0.01"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.6 0.6 0.6"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="10 10 0.1" rgba="0.25 0.27 0.30 1"/>
{drones_xml}
{payload_xml}
  </worldbody>
</mujoco>'''


def _transport_pieces(plan, origin=(0.0, 0.0), tether_len=1.0, drone_model="x2"):
    """Build the payload + carrier-drone + grip-constraint XML for a transport job
    at a given xy ``origin``.

    The payload is a thin slab of square tiles, one per cell; a drone hovers over
    each tile's CENTER with a vertical suction cable straight down. The tile edge
    is the drone-to-drone spacing, so a sized-up slab gives a safe gap directly.
    Returns (payload_xml, drones_xml, eqs_xml, info)."""
    shape = plan["shape"]
    tile_half = plan["cube_edge"] / 2
    half_z = plan["thickness"] / 2
    ox0, oy0 = origin
    payload_z = half_z                          # slab rests on the ground
    payload_top = 2 * half_z
    contact_z = payload_top + tether_len        # drone center when the cup touches

    cells = cell_offsets(shape, tile_half)
    grips = np.array(grip_points(cells, plan["n_drones"], tile_half))
    # nearest-neighbour spacing == safe gap (tile edge, since one drone per tile)
    if len(grips) > 1:
        d = np.linalg.norm(grips[:, None, :] - grips[None, :, :], axis=-1)
        d[d == 0] = np.inf
        min_sep = float(d.min())
    else:
        min_sep = float("inf")

    payload_xml = _payload_body(shape, spawn=(ox0, oy0, payload_z),
                                cube_mass=plan["mass_per_cube"], cube_half=tile_half,
                                half_z=half_z)

    drones, eqs = [], []
    vec = (0.0, 0.0, -tether_len)                # straight down from drone center
    for i, (gx, gy) in enumerate(grips):
        drones.append(_drone_body(i, (ox0 + gx, oy0 + gy, contact_z),
                                  model=drone_model, tether=vec))
        # Grip at the cup tip; inactive until the drone descends onto the tile,
        # then switched on (suction) at runtime via data.eq_active.
        eqs.append(f'    <connect name="grip{i}" body1="drone{i}" body2="payload" '
                   f'anchor="0 0 {-tether_len}" active="false"/>')

    info = {
        "shape": shape,
        "n_carriers": len(grips),
        "offsets": [tuple(g) for g in grips],   # drone xy = tile centers
        "min_drone_sep": min_sep,
        "origin": (ox0, oy0),
        "payload_z": payload_z,
        "payload_top": payload_top,
        "contact_z": contact_z,
        "tether_len": tether_len,
        "payload_mass": plan["payload_mass"],
        "share_per_drone": plan["share_per_drone"],
        "cube_half": tile_half,
        "half_z": half_z,
    }
    return payload_xml, "\n".join(drones), "\n".join(eqs), info


def build_transport_scene(payload_shape="Z", plan=None, tether_len=1.0,
                          drone_model="x2", **plan_kwargs):
    """Scene for Layer 2 — cooperative transport with suction-cup pickup.

    Uses ``plan_transport`` to size the job: the payload mass comes from the cube
    size + density, and only the *necessary* number of carriers is spawned. Each
    drone hangs a string with a suction cup; it spawns at the contact pose with
    its grip constraint inactive. The controller then runs approach -> descend ->
    suction on -> lift -> carry (see scripts/demo_transport.py).

    Returns ``(xml, info)``; ``info`` includes the full ``plan`` so callers can
    report the weight and drone count.
    """
    plan = plan or plan_transport(shape=payload_shape, **plan_kwargs)
    payload_xml, drones_xml, eqs_xml, info = _transport_pieces(
        plan, tether_len=tether_len, drone_model=drone_model)
    assets = _x2_asset_block() if drone_model == "x2" else ""
    info["plan"] = plan

    xml = f'''<mujoco model="tetraswarm_transport">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.6 0.6 0.6"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="10 10 0.1" rgba="0.25 0.27 0.30 1"/>
{drones_xml}
{payload_xml}
  </worldbody>
  <equality>
{eqs_xml}
  </equality>
</mujoco>'''
    return xml, info


def _wall(name, center, half, rgba="0.42 0.44 0.50 1", solid=False):
    # Non-colliding by default: walls are visual guides the swarm threads by
    # planning (avoids the big slab pinning itself on a boundary). When
    # ``solid=True`` the wall physically collides (e.g. the squeeze doorway, so
    # the payload is genuinely blocked unless it turns to fit).
    cx, cy, cz = center
    hx, hy, hz = half
    col = "" if solid else ' contype="0" conaffinity="0"'
    return (f'    <geom name="{name}" type="box" pos="{cx} {cy} {cz}" '
            f'size="{hx} {hy} {hz}"{col} rgba="{rgba}"/>')


def _arena_layout(arena=(9.0, 6.0), wall_h=2.0, thick=0.15, gate_x=2.5, overlap=0.5):
    """Walls + weave waypoints for the slalom arena. Shared by the navigation
    scene and the combined mission scene. The arena is sized generously so a
    well-spaced formation (drones ~1 m+ apart) clears every gap with margin.
    Returns (walls_xml, waypoints, start, goal)."""
    ax, ay = arena
    start = (-ax + 3.0, 0.0)                       # inset so a big slab clears the wall
    goal = (ax - 3.0, 0.0)
    # Gate A blocks the lower band y in [-ay, overlap] (gap above); gate B blocks
    # the upper band y in [-overlap, ay] (gap below) -> each gap is ~ay-overlap m.
    a_lo, a_hi = -ay, overlap
    b_lo, b_hi = -overlap, ay
    walls = [
        _wall("w_n", (0, ay, wall_h), (ax, thick, wall_h), solid=True),
        _wall("w_s", (0, -ay, wall_h), (ax, thick, wall_h), solid=True),
        _wall("w_e", (ax, 0, wall_h), (thick, ay, wall_h), solid=True),
        _wall("w_w", (-ax, 0, wall_h), (thick, ay, wall_h), solid=True),
        _wall("gate_a", (-gate_x, (a_lo + a_hi) / 2, wall_h),
              (thick, (a_hi - a_lo) / 2, wall_h), rgba="0.55 0.35 0.35 1", solid=True),
        _wall("gate_b", (gate_x, (b_lo + b_hi) / 2, wall_h),
              (thick, (b_hi - b_lo) / 2, wall_h), rgba="0.35 0.45 0.55 1", solid=True),
    ]
    # Centre of each gap, with lead-in/lead-out waypoints so the formation aims
    # straight through rather than cutting the wall ends.
    gap_a_y = (overlap + ay) / 2          # centre of upper gap
    gap_b_y = -(overlap + ay) / 2         # centre of lower gap
    waypoints = [(-gate_x - 1.5, gap_a_y), (-gate_x, gap_a_y),
                 (gate_x, gap_b_y), (gate_x + 1.5, gap_b_y), goal]
    return "\n".join(walls), waypoints, start, goal


def build_navigation_scene(n_drones=6, drone_model="x2", nav_z=1.5,
                           arena=(9.0, 6.0), wall_h=2.0, thick=0.15):
    """Layer-5 scene — a walled slalom the swarm must coordinate through.

    Two offset gate walls force the swarm to weave (up through gap A, down
    through gap B) on its way from the start zone to the goal, so it has to stay
    in a compact formation and navigate, not just fly straight.

    Returns ``(xml, info)`` with the start, goal and gate waypoints for the
    navigation controller.
    """
    walls_xml, waypoints, start, goal = _arena_layout(arena, wall_h, thick)

    # Spawn drones already in the formation ring (radius matches demo slot_offsets)
    # so they don't cross paths flying into formation and collide at t=0.
    drones = []
    ring = 1.15
    for i in range(n_drones):
        a = 2 * np.pi * i / n_drones
        px = start[0] + ring * np.cos(a)
        py = start[1] + ring * np.sin(a)
        drones.append(_drone_body(i, (px, py, nav_z), model=drone_model))
    drones_xml = "\n".join(drones)
    assets = _x2_asset_block() if drone_model == "x2" else ""

    xml = f'''<mujoco model="tetraswarm_nav">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <default>
    <geom solref="0.02 1" solimp="0.8 0.9 0.01"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="12 12 0.1" rgba="0.23 0.25 0.28 1"/>
    <site name="goal" pos="{goal[0]} {goal[1]} {nav_z}" size="0.35" rgba="0.2 0.9 0.3 0.4"/>
{walls_xml}
{drones_xml}
  </worldbody>
</mujoco>'''

    info = {
        "n_drones": n_drones,
        "start": start,
        "goal": goal,
        "waypoints": waypoints,
        "nav_z": nav_z,
        "arena": arena,
    }
    return xml, info


def payload_extent(info):
    """Footprint half-extents (x, y) of the payload from its grip layout + tile."""
    g = np.array(info["offsets"])
    th = info["cube_half"]
    return float(np.abs(g[:, 0]).max() + th), float(np.abs(g[:, 1]).max() + th)


def build_squeeze_scene(payload_shape="Z", plan=None, tether_len=1.0,
                        drone_model="x2", gap_w=3.5, span=9.0, start_y=-6.0,
                        goal_y=6.0, wall_y=0.0, wall_h=2.0, thick=0.15):
    """Scene for the 'turn-to-fit' maneuver. A wall across the path (running along
    x at ``wall_y``) has a gap only ``gap_w`` wide — narrower than the Cleveland Z
    is across — so the swarm must rotate the slab 90 deg to slip its narrow side
    through, then rotate back. The swarm travels in +y from start to goal.

    Returns ``(xml, info)`` with start/goal and the gap so the controller can plan
    the turn."""
    plan = plan or plan_transport(shape=payload_shape)
    payload_xml, drones_xml, eqs_xml, info = _transport_pieces(
        plan, origin=(0.0, start_y), tether_len=tether_len, drone_model=drone_model)
    assets = _x2_asset_block() if drone_model == "x2" else ""

    hw = gap_w / 2.0
    seg = (span - hw) / 2.0                       # half-length of each wall stub
    walls = "\n".join([
        _wall("door_l", (-(hw + seg), wall_y, wall_h), (seg, thick, wall_h),
              rgba="0.55 0.4 0.35 1", solid=True),
        _wall("door_r", (hw + seg, wall_y, wall_h), (seg, thick, wall_h),
              rgba="0.55 0.4 0.35 1", solid=True),
    ])
    info.update({"plan": plan, "start": (0.0, start_y), "goal": (0.0, goal_y),
                 "gap_w": gap_w, "wall_y": wall_y})

    xml = f'''<mujoco model="tetraswarm_squeeze">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <default>
    <geom solref="0.02 1" solimp="0.8 0.9 0.01"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="14 14 0.1" rgba="0.23 0.25 0.28 1"/>
    <site name="goal" pos="0 {goal_y} 1.0" size="0.35" rgba="0.2 0.9 0.3 0.4"/>
{walls}
{drones_xml}
{payload_xml}
  </worldbody>
  <equality>
{eqs_xml}
  </equality>
</mujoco>'''
    return xml, info


def _gen_braided_maze(nx, ny, seed=7):
    """Perfect maze via randomized DFS, then 'braided' (every dead-end gets one
    extra opening) so there are NO dead-ends — every cell has >=2 connections and
    loops exist. Returns the set of open connections between adjacent cells."""
    import random
    rng = random.Random(seed)
    nbrs = lambda c: [(c[0] + dx, c[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                      if 0 <= c[0] + dx < nx and 0 <= c[1] + dy < ny]
    visited, conn, stack = {(0, 0)}, set(), [(0, 0)]
    while stack:
        c = stack[-1]
        opts = [n for n in nbrs(c) if n not in visited]
        if opts:
            n = rng.choice(opts)
            conn.add(frozenset((c, n)))
            visited.add(n)
            stack.append(n)
        else:
            stack.pop()
    for i in range(nx):                           # braid: kill dead-ends
        for j in range(ny):
            c = (i, j)
            deg = sum(c in k for k in conn)
            if deg <= 1:
                extra = [n for n in nbrs(c) if frozenset((c, n)) not in conn]
                if extra:
                    conn.add(frozenset((c, rng.choice(extra))))
    return conn


def _maze_route(conn, start, goal):
    """Shortest path (BFS) start->goal cell through the open connections."""
    from collections import deque
    q, prev = deque([start]), {start: None}
    while q:
        c = q.popleft()
        if c == goal:
            break
        for k in conn:
            if c in k:
                n = next(iter(k - {c}))
                if n not in prev:
                    prev[n] = c
                    q.append(n)
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev.get(c)
    return path[::-1]


def build_maze_scene(nx=5, ny=5, cell=9.0, seed=7, plan=None, tether_len=1.0,
                     drone_model="x2", wall_h=2.0, thick=0.15):
    """A braided (no-dead-end) maze of corridors with doorway gaps of varied
    width. The payload travels the BFS route start->goal; some gaps are narrower
    than the Z is wide, so the couriers must keep its narrow side facing each gap
    (yaw = travel heading). Returns ``(xml, info)`` with the route + gap list."""
    import random
    plan = plan or plan_transport(shape="Z")
    conn = _gen_braided_maze(nx, ny, seed)
    rng = random.Random(seed + 1)
    # Doorway width per opening: some full-open, some narrow enough to force turns.
    # mixed widths: some full-open cells, some narrower doorways (>=4.5 m so any
    # tetromino's short side, up to ~3.8 m for L, clears with margin)
    widths = {k: rng.choice([4.5, 6.0, cell, cell]) for k in conn}

    x0, y0 = -nx * cell / 2.0, -ny * cell / 2.0
    center = lambda c: (x0 + (c[0] + 0.5) * cell, y0 + (c[1] + 0.5) * cell)
    walls = []

    def vwall(name, x, ylo, yhi):                 # vertical wall segment (solid)
        walls.append(_wall(name, (x, (ylo + yhi) / 2, wall_h),
                           (thick, (yhi - ylo) / 2, wall_h), solid=True))

    def hwall(name, y, xlo, xhi):                 # horizontal wall segment (solid)
        walls.append(_wall(name, ((xlo + xhi) / 2, y, wall_h),
                           ((xhi - xlo) / 2, thick, wall_h), solid=True))

    # outer boundary
    vwall("b_w", x0, y0, y0 + ny * cell)
    vwall("b_e", x0 + nx * cell, y0, y0 + ny * cell)
    hwall("b_s", y0, x0, x0 + nx * cell)
    hwall("b_n", y0 + ny * cell, x0, x0 + nx * cell)

    # interior edges: solid wall if not connected, else a centered doorway gap
    for i in range(nx):
        for j in range(ny):
            c = (i, j)
            if i + 1 < nx:                        # edge to the east neighbour
                x = x0 + (i + 1) * cell
                ylo, yhi = y0 + j * cell, y0 + (j + 1) * cell
                k = frozenset((c, (i + 1, j)))
                if k not in conn:
                    vwall(f"v{i}_{j}", x, ylo, yhi)
                elif widths[k] < cell:            # doorway: two stubs, gap in middle
                    g = widths[k]
                    ymid = (ylo + yhi) / 2
                    vwall(f"v{i}_{j}a", x, ylo, ymid - g / 2)
                    vwall(f"v{i}_{j}b", x, ymid + g / 2, yhi)
            if j + 1 < ny:                        # edge to the north neighbour
                y = y0 + (j + 1) * cell
                xlo, xhi = x0 + i * cell, x0 + (i + 1) * cell
                k = frozenset((c, (i, j + 1)))
                if k not in conn:
                    hwall(f"h{i}_{j}", y, xlo, xhi)
                elif widths[k] < cell:
                    g = widths[k]
                    xmid = (xlo + xhi) / 2
                    hwall(f"h{i}_{j}a", y, xlo, xmid - g / 2)
                    hwall(f"h{i}_{j}b", y, xmid + g / 2, xhi)

    start_cell, goal_cell = (0, 0), (nx - 1, ny - 1)
    route_cells = _maze_route(conn, start_cell, goal_cell)
    route = [center(c) for c in route_cells]
    start, goal = route[0], route[-1]

    payload_xml, drones_xml, eqs_xml, info = _transport_pieces(
        plan, origin=start, tether_len=tether_len, drone_model=drone_model)
    assets = _x2_asset_block() if drone_model == "x2" else ""
    # Scout coverage path: a DFS over the OPEN connections (through doorways), so
    # the scout visits every cell while staying in the corridors — it never flies
    # through a wall. Backtracking steps retrace the corridor it came in by.
    cover_cells, seen = [], set()

    def _cover(c):
        seen.add(c)
        cover_cells.append(c)
        for k in conn:
            if c in k:
                nb = next(iter(k - {c}))
                if nb not in seen:
                    _cover(nb)
                    cover_cells.append(c)        # retrace back through the doorway
    _cover(start_cell)
    sweep = [center(c) for c in cover_cells]
    info.update({"plan": plan, "route": route, "start": start, "goal": goal,
                 "scout_z": 2.8, "cell": cell, "nx": nx, "ny": ny, "sweep": sweep})

    # "Viper" scout: a lone recon drone flying AHEAD of the couriers (not on top
    # of the payload). Non-colliding so it can range freely to map.
    spx, spy = route[1] if len(route) > 1 else start
    scout = f'''    <body name="scout" pos="{spx} {spy} {info['scout_z']}">
      <freejoint name="scout_free"/>
      <geom type="mesh" mesh="X2" material="x2mat" quat="0 0 1 1" contype="0" conaffinity="0" mass="0" group="2"/>
      <geom type="box" size="0.18 0.18 0.04" mass="0.3" contype="0" conaffinity="0" rgba="0.9 0.15 0.15 1" group="3"/>
    </body>'''

    xml = f'''<mujoco model="tetraswarm_maze">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <default>
    <geom solref="0.02 1" solimp="0.8 0.9 0.01"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="26 26 0.1" rgba="0.23 0.25 0.28 1"/>
    <site name="goal" pos="{goal[0]} {goal[1]} 1.0" size="0.35" rgba="0.2 0.9 0.3 0.4"/>
{chr(10).join(walls)}
{scout}
{drones_xml}
{payload_xml}
  </worldbody>
  <equality>
{eqs_xml}
  </equality>
</mujoco>'''
    return xml, info


def build_mission_scene(payload_shape="Z", plan=None, tether_len=1.0,
                        drone_model="x2", arena=(9.0, 6.0), wall_h=2.0,
                        thick=0.15, **plan_kwargs):
    """Combined Layer 2 + 5 — pick up a payload at the start zone and navigate it
    through the walled slalom to the goal. Walls from the nav arena + carriers,
    payload and suction grips from the transport job, in one scene.

    Returns ``(xml, info)``; ``info`` merges the transport layout with the arena
    ``waypoints``/``start``/``goal`` so the controller can grip then weave.
    """
    plan = plan or plan_transport(shape=payload_shape, **plan_kwargs)
    walls_xml, waypoints, start, goal = _arena_layout(arena, wall_h, thick)
    payload_xml, drones_xml, eqs_xml, info = _transport_pieces(
        plan, origin=start, tether_len=tether_len, drone_model=drone_model)
    assets = _x2_asset_block() if drone_model == "x2" else ""
    info.update({"plan": plan, "waypoints": waypoints, "start": start, "goal": goal,
                 "arena": arena, "scout_z": 2.8})

    # "Viper" scout: a lone recon drone (red) that flies the route first to map
    # it, so the couriers can plan in an unknown environment with no top camera.
    sx, sy = start
    scout = f'''    <body name="scout" pos="{sx} {sy - 1.5} {info['scout_z']}">
      <freejoint name="scout_free"/>
      <geom type="mesh" mesh="X2" material="x2mat" quat="0 0 1 1" contype="0" conaffinity="0" mass="0" group="2"/>
      <geom type="box" size="0.18 0.18 0.04" mass="0.3" contype="0" conaffinity="0" rgba="0.9 0.15 0.15 1" group="3"/>
    </body>'''

    xml = f'''<mujoco model="tetraswarm_mission">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <default>
    <geom solref="0.02 1" solimp="0.8 0.9 0.01"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="12 12 0.1" rgba="0.23 0.25 0.28 1"/>
    <site name="goal" pos="{goal[0]} {goal[1]} 1.0" size="0.35" rgba="0.2 0.9 0.3 0.4"/>
{walls_xml}
{scout}
{drones_xml}
{payload_xml}
  </worldbody>
  <equality>
{eqs_xml}
  </equality>
</mujoco>'''
    return xml, info


def _free_block_body(name, shape, pos, tile_half, half_z, color, yaw=0.0):
    """A free (kinematically-driven) tetromino slab block for the relay/mission
    scene. ``yaw`` rotates the body about z at spawn (lets the dock pack each block
    on its narrow side)."""
    geoms = []
    for (x, y) in cell_offsets(shape, tile_half):
        geoms.append(                              # SOLID (collides with walls)
            f'      <geom type="box" size="{tile_half} {tile_half} {half_z}" '
            f'pos="{x:.3f} {y:.3f} 0" rgba="{color}"/>')
    sx, sy, sz = pos
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    quat = f' quat="{qw:.6f} 0 0 {qz:.6f}"' if abs(yaw) > 1e-9 else ""
    return f'''    <body name="{name}" pos="{sx} {sy} {sz}"{quat}>
      <freejoint name="{name}_free"/>
{chr(10).join(geoms)}
    </body>'''


def build_relay_scene(tile_edge=1.0, thickness=0.08, tether_len=0.8, drone_model="x2",
                      arena=13.0):
    """Warehouse relay scene: 4 tetromino blocks (Z, I, O, L) tetris-aligned at a
    central depot, a pinwheel of turning walls, and 4 corner drop zones. The
    blocks are kinematically carried (their pose is driven to follow the carrier
    formation), so the swarm can pick up / carry / drop each in turn. Returns
    ``(xml, info)`` with block layouts, depot poses and corner targets."""
    th, hz = tile_edge / 2, thickness / 2
    blocks = ["Z", "I", "O", "L"]
    colors = {"Z": "0.85 0.4 0.4 1", "I": "0.4 0.7 0.9 1",
              "O": "0.95 0.8 0.3 1", "L": "0.6 0.5 0.85 1"}
    depot = {"Z": (-3.5, -3.5), "I": (3.5, -3.5), "O": (-3.5, 3.5), "L": (3.5, 3.5)}
    corners = {"top_left": (-arena + 2.5, arena - 2.5),
               "top_right": (arena - 2.5, arena - 2.5),
               "bottom_right": (arena - 2.5, -arena + 2.5),
               "bottom_left": (-arena + 2.5, -arena + 2.5)}
    # delivery: Z->TL, I->BR, O->TR, L->BL
    deliver = {"Z": "top_left", "I": "bottom_right", "O": "top_right", "L": "bottom_left"}

    a, t = arena, 0.15
    wh = 2.0
    walls = [_wall("b_n", (0, a, wh), (a, t, wh), solid=True),
             _wall("b_s", (0, -a, wh), (a, t, wh), solid=True),
             _wall("b_e", (a, 0, wh), (t, a, wh), solid=True),
             _wall("b_w", (-a, 0, wh), (t, a, wh), solid=True)]
    # Pinwheel: 4 offset walls so the path from the depot to each corner must
    # turn (no straight shot across the arena).
    pin = [(-7, 1.5, 1.5, t), (7, -1.5, 1.5, t),   # horizontal stubs (cx,cy,hx,hy)
           (1.5, 7, t, 1.5), (-1.5, -7, t, 1.5)]   # vertical stubs
    for i, (cx, cy, hx, hy) in enumerate(pin):
        walls.append(_wall(f"pin{i}", (cx, cy, wh), (hx, hy, wh),
                           rgba="0.5 0.45 0.4 1", solid=True))

    drones = "\n".join(_drone_body(i, (depot["Z"][0] + cell_offsets("Z", th)[i][0],
                                       depot["Z"][1] + cell_offsets("Z", th)[i][1],
                                       hz * 2 + tether_len),
                                   model=drone_model, tether=(0, 0, -tether_len),
                                   collide=False)
                       for i in range(4))
    block_xml = "\n".join(_free_block_body(s, s, (*depot[s], hz), th, hz, colors[s])
                          for s in blocks)
    sites = "\n".join(
        f'    <site name="goal_{c}" pos="{x} {y} 0.05" size="0.4" '
        f'rgba="0.2 0.9 0.3 0.35"/>' for c, (x, y) in corners.items())
    assets = _x2_asset_block() if drone_model == "x2" else ""

    xml = f'''<mujoco model="tetraswarm_relay">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.45 0.45 0.45"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="20 20 0.1" rgba="0.24 0.26 0.29 1"/>
{sites}
{chr(10).join(walls)}
{drones}
{block_xml}
  </worldbody>
</mujoco>'''

    info = {"blocks": blocks, "depot": depot, "corners": corners, "deliver": deliver,
            "grips": {s: cell_offsets(s, th) for s in blocks},
            "tile_half": th, "half_z": hz, "tether_len": tether_len, "colors": colors}
    return xml, info


def _gen_maze_walls(nx, ny, cell, conn, widths, wall_h, thick):
    """Shared maze-wall builder -> (walls_xml, x0, y0, center_fn)."""
    x0, y0 = -nx * cell / 2.0, -ny * cell / 2.0
    center = lambda c: (x0 + (c[0] + 0.5) * cell, y0 + (c[1] + 0.5) * cell)
    walls = []

    def vwall(name, x, ylo, yhi):
        walls.append(_wall(name, (x, (ylo + yhi) / 2, wall_h),
                           (thick, (yhi - ylo) / 2, wall_h), solid=True))

    def hwall(name, y, xlo, xhi):
        walls.append(_wall(name, ((xlo + xhi) / 2, y, wall_h),
                           ((xhi - xlo) / 2, thick, wall_h), solid=True))

    vwall("b_w", x0, y0, y0 + ny * cell)
    vwall("b_e", x0 + nx * cell, y0, y0 + ny * cell)
    hwall("b_s", y0, x0, x0 + nx * cell)
    hwall("b_n", y0 + ny * cell, x0, x0 + nx * cell)
    for i in range(nx):
        for j in range(ny):
            c = (i, j)
            if i + 1 < nx:
                x = x0 + (i + 1) * cell
                ylo, yhi = y0 + j * cell, y0 + (j + 1) * cell
                k = frozenset((c, (i + 1, j)))
                if k not in conn:
                    vwall(f"v{i}_{j}", x, ylo, yhi)
                elif widths[k] < cell:
                    g = widths[k]; ym = (ylo + yhi) / 2
                    vwall(f"v{i}_{j}a", x, ylo, ym - g / 2)
                    vwall(f"v{i}_{j}b", x, ym + g / 2, yhi)
            if j + 1 < ny:
                y = y0 + (j + 1) * cell
                xlo, xhi = x0 + i * cell, x0 + (i + 1) * cell
                k = frozenset((c, (i, j + 1)))
                if k not in conn:
                    hwall(f"h{i}_{j}", y, xlo, xhi)
                elif widths[k] < cell:
                    g = widths[k]; xm = (xlo + xhi) / 2
                    hwall(f"h{i}_{j}a", y, xlo, xm - g / 2)
                    hwall(f"h{i}_{j}b", y, xm + g / 2, xhi)
    return "\n".join(walls), x0, y0, center


# ---- Farama Gymnasium-Robotics PointMaze-style maze maps ----
# 1 = wall, 0 = free, "g" = goal/drop-zone, "r" = reset/start. (robotics.farama.org)
POINTMAZE_LARGE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, "g", 0, 0, 0, "g", 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, "r", 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, "g", 0, 1, 0, 0, 0, 1, 0, 0, "g", 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def maze_from_grid(nx=10, ny=8, seed=0, n_goals=4, mission=False):
    """Generate an UNKNOWN PointMaze-format map (1=wall,0=free,'g'=goal,'r'=start)
    by carving a perfect maze with DFS on a cell grid, then a few loops removed
    for braiding. Walls are full grid cells (PointMaze style). With ``mission`` the
    start (dock) is the BOTTOM-LEFT cell and a single goal sits TOP-RIGHT (the far
    end the blocks must reach)."""
    import random
    rng = random.Random(seed)
    W, H = 2 * nx + 1, 2 * ny + 1
    g = [[1] * W for _ in range(H)]
    def carve(cx, cy):
        g[2 * cy + 1][2 * cx + 1] = 0
        for dx, dy in rng.sample([(1, 0), (-1, 0), (0, 1), (0, -1)], 4):
            nxc, nyc = cx + dx, cy + dy
            if 0 <= nxc < nx and 0 <= nyc < ny and g[2 * nyc + 1][2 * nxc + 1] == 1:
                g[2 * cy + 1 + dy][2 * cx + 1 + dx] = 0
                carve(nxc, nyc)
    carve(nx // 2, ny // 2)
    for _ in range(nx * ny // 4):                     # braid: knock out a few walls
        i, j = rng.randrange(1, H - 1), rng.randrange(1, W - 1)
        if g[i][j] == 1 and ((g[i - 1][j] == 0 and g[i + 1][j] == 0) or
                             (g[i][j - 1] == 0 and g[i][j + 1] == 0)):
            g[i][j] = 0
    free = [(i, j) for i in range(H) for j in range(W) if g[i][j] == 0]
    if mission:
        sc = min(free, key=lambda c: c[1] - c[0])     # bottom-left (small col, big row)
        dc = min(free, key=lambda c: c[0] - c[1])     # top-right  (small row, big col)
        g[sc[0]][sc[1]] = "r"
        g[dc[0]][dc[1]] = "g"
    else:
        g[2 * (ny // 2) + 1][2 * (nx // 2) + 1] = "r"  # start at centre
        corners = sorted(free, key=lambda c: -(abs(c[0] - H / 2) + abs(c[1] - W / 2)))
        for c in corners[:n_goals]:                   # goals at the far cells
            g[c[0]][c[1]] = "g"
    return g


def build_pointmaze_scene(maze_map=None, cell=5.0, wall_h=2.0, scout_z=1.0,
                          n_drones=4, n_rays=24, drone_model="x2", drone_size=0.16,
                          dock_blocks=None, block_tile=0.4):
    """Scene from a Farama PointMaze-format map (full-cell wall blocks). The maze is
    UNKNOWN to the swarm: drones explore + map it and DISCOVER the goal/drop-zone
    cells ('g'). Returns ``(xml, info)`` with goal world positions + ground-truth
    free grid for scoring."""
    if maze_map is None:
        maze_map = POINTMAZE_LARGE
    nrows, ncols = len(maze_map), len(maze_map[0])
    x0, y0 = -ncols * cell / 2.0, -nrows * cell / 2.0
    center = lambda i, j: (x0 + (j + 0.5) * cell, y0 + (nrows - 1 - i + 0.5) * cell)

    walls, goals, start = [], [], None
    free_grid = np.zeros((nrows, ncols), bool)
    for i in range(nrows):
        for j in range(ncols):
            v = maze_map[i][j]
            cx, cy = center(i, j)
            if v == 1:
                walls.append(_wall(f"w{i}_{j}", (cx, cy, wall_h),
                                   (cell / 2, cell / 2, wall_h), solid=True))
            else:
                free_grid[i, j] = True
                if v == "g":
                    goals.append((cx, cy))
                elif v in ("r", "c"):
                    start = (cx, cy)
    if start is None:
        fi = np.argwhere(free_grid)
        start = center(*fi[len(fi) // 2])

    sites = "\n".join(
        f'    <site name="goal{k}" pos="{gx} {gy} 0.15" size="{cell * 0.12:.2f}" '
        f'rgba="0.35 0.35 0.42 0.06"/>' for k, (gx, gy) in enumerate(goals))

    angles = [2 * np.pi * i / n_rays for i in range(n_rays)] if n_rays else []
    cols = ["0.9 0.15 0.15 1", "0.2 0.55 0.95 1", "0.2 0.8 0.4 1", "0.95 0.75 0.2 1"]
    drones = []
    for d in range(n_drones):
        rays = "\n".join(
            f'        <site name="ray{d}_{i}" pos="0 0 0" '
            f'zaxis="{np.cos(a):.4f} {np.sin(a):.4f} 0"/>' for i, a in enumerate(angles))
        ox, oy = start[0] + (d % 2 - 0.5) * 1.2, start[1] + (d // 2 - 0.5) * 1.2
        drones.append(f'''    <body name="scout{d}" pos="{ox} {oy} {scout_z}">
      <freejoint name="scout{d}_free"/>
      <geom type="mesh" mesh="X2" material="x2mat" quat="0 0 1 1" contype="0" conaffinity="0" mass="0" group="2"/>
      <geom type="box" size="{drone_size} {drone_size} {drone_size * 0.25:.3f}" mass="0.3" rgba="{cols[d % 4]}" group="3"/>
{rays}
    </body>''')
    sensors = "\n".join(f'    <rangefinder name="r{d}_{i}" site="ray{d}_{i}"/>'
                        for d in range(n_drones) for i in range(n_rays))
    assets = _x2_asset_block() if drone_model == "x2" else ""

    # ---- DOCK PAYLOAD: Tetris-coloured tetromino blocks packed at the start cell.
    # Each block is oriented on whichever side (0 or 90 deg) is narrower, then laid
    # out in a single row centred on the dock and spaced so footprints never
    # overlap — and so the whole row stays inside the open 'r' cell (no wall poke).
    block_bodies, block_info, th, hz = [], {}, block_tile, 0.15
    if dock_blocks:
        def _extent(shape, yaw):
            offs = np.array(cell_offsets(shape, th))
            c, s = np.cos(yaw), np.sin(yaw)
            r = offs @ np.array([[c, -s], [s, c]]).T
            return np.ptp(r[:, 0]) + 2 * th, np.ptp(r[:, 1]) + 2 * th
        yaws, widths = {}, {}
        for sh in dock_blocks:
            w0, _ = _extent(sh, 0.0)
            w9, _ = _extent(sh, np.pi / 2)
            yaws[sh] = 0.0 if w0 <= w9 else np.pi / 2
            widths[sh] = min(w0, w9)
        gap = 0.45
        total = sum(widths[s] for s in dock_blocks) + gap * (len(dock_blocks) - 1)
        x = start[0] - total / 2.0
        for sh in dock_blocks:
            cx = x + widths[sh] / 2.0
            x += widths[sh] + gap
            block_bodies.append(_free_block_body(sh, sh, (cx, start[1], hz), th, hz,
                                                 TETRIS_COLORS[sh], yaw=yaws[sh]))
            block_info[sh] = {"pos": (cx, start[1]), "yaw": yaws[sh],
                              "grips": cell_offsets(sh, th), "color": TETRIS_COLORS[sh]}
    blocks_xml = "\n".join(block_bodies)

    xml = f'''<mujoco model="tetraswarm_pointmaze">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <default><geom solref="0.02 1" solimp="0.8 0.9 0.01"/></default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0 0 0" ambient="0.65 0.65 0.65" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light name="top" dir="0 0 -1" directional="true" diffuse="0.55 0.55 0.55"/>
    <geom name="ground" type="plane" size="80 80 0.1" rgba="0.08 0.09 0.11 1"/>
{sites}
{chr(10).join(walls)}
{chr(10).join(drones)}
{blocks_xml}
  </worldbody>
  <sensor>
{sensors}
  </sensor>
</mujoco>'''

    info = {"nrows": nrows, "ncols": ncols, "cell": cell, "x0": x0, "y0": y0,
            "center": start, "goals": goals, "free_grid": free_grid,
            "n_drones": n_drones, "n_rays": n_rays, "angles": angles,
            "scout_z": scout_z, "maze_map": maze_map,
            "blocks": block_info, "dock_blocks": list(dock_blocks or []),
            "block_tile": th, "block_hz": hz}
    return xml, info


def build_scout_scene(nx=7, ny=7, cell=7.5, seed=11, n_rays=24, scout_z=1.5,
                      n_drones=4, drone_model="x2", drone_size=0.16, tile_edge=0.8):
    """Scout/sweep scene on the SAME maze as build_maze_relay_scene (identical
    braided layout + narrow doorways), with ``n_drones`` lidar drones. Used to
    export the maze for ViPER and to replay sweeps. Returns ``(xml, info)``."""
    import random
    # reproduce the relay maze EXACTLY (same conn, widths, narrow I/L doorways)
    conn = _gen_braided_maze(nx, ny, seed)
    rng = random.Random(seed + 1)
    widths = {k: rng.choice([4.5, 6.0, cell, cell]) for k in conn}
    depot_cell = (nx // 2, ny // 2)
    corner_cells = {"bottom_left": (0, 0), "bottom_right": (nx - 1, 0),
                    "top_left": (0, ny - 1), "top_right": (nx - 1, ny - 1)}
    route_cells = {name: _maze_route(conn, depot_cell, cc)
                   for name, cc in corner_cells.items()}
    narrow = 1.5 * 2 * tile_edge
    for name in ("bottom_right", "bottom_left"):
        rc = route_cells[name]
        for ca, cb in zip(rc, rc[1:]):
            widths[frozenset((ca, cb))] = narrow
    walls_xml, x0, y0, center = _gen_maze_walls(nx, ny, cell, conn, widths, 2.0, 0.15)

    angles = [2 * np.pi * i / n_rays for i in range(n_rays)]
    cx, cy = center((nx // 2, ny // 2))
    cols = ["0.9 0.15 0.15 1", "0.2 0.55 0.95 1", "0.2 0.8 0.4 1", "0.95 0.75 0.2 1"]
    drones = []
    for d in range(n_drones):
        rays = "\n".join(
            f'        <site name="ray{d}_{i}" pos="0 0 0" '
            f'zaxis="{np.cos(a):.4f} {np.sin(a):.4f} 0"/>' for i, a in enumerate(angles))
        ox, oy = cx + (d % 2 - 0.5) * 1.4, cy + (d // 2 - 0.5) * 1.4
        # mass box is SOLID (collides with walls + other drones); visual mesh isn't
        drones.append(f'''    <body name="scout{d}" pos="{ox} {oy} {scout_z}">
      <freejoint name="scout{d}_free"/>
      <geom type="mesh" mesh="X2" material="x2mat" quat="0 0 1 1" contype="0" conaffinity="0" mass="0" group="2"/>
      <geom type="box" size="{drone_size} {drone_size} {drone_size*0.25:.3f}" mass="0.3" rgba="{cols[d % 4]}" group="3"/>
{rays}
    </body>''')
    sensors = "\n".join(f'    <rangefinder name="r{d}_{i}" site="ray{d}_{i}"/>'
                        for d in range(n_drones) for i in range(n_rays))
    assets = _x2_asset_block() if drone_model == "x2" else ""

    xml = f'''<mujoco model="tetraswarm_scout">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <default>
    <geom solref="0.02 1" solimp="0.8 0.9 0.01"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0 0 0" ambient="0.65 0.65 0.65" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light name="top" dir="0 0 -1" directional="true" diffuse="0.55 0.55 0.55"/>
    <geom name="ground" type="plane" size="40 40 0.1" rgba="0.08 0.09 0.11 1"/>
{walls_xml}
{chr(10).join(drones)}
  </worldbody>
  <sensor>
{sensors}
  </sensor>
</mujoco>'''

    info = {"nx": nx, "ny": ny, "cell": cell, "x0": x0, "y0": y0, "seed": seed,
            "conn": conn, "angles": angles, "n_rays": n_rays, "scout_z": scout_z,
            "n_drones": n_drones, "center": (cx, cy)}
    return xml, info


def build_maze_relay_scene(nx=7, ny=7, cell=7.5, seed=11, tile_edge=0.8,
                           thickness=0.08, tether_len=0.8, drone_model="x2",
                           wall_h=2.0, thick=0.15):
    """Multi-block relay INSIDE a braided maze. Four tetromino blocks sit at the
    centre cell (depot); the four maze corners are the drop zones. Every delivery
    is routed through the maze (BFS) — there is no straight or diagonal shortcut
    between corners. Returns ``(xml, info)`` with per-corner routes + layouts."""
    conn = _gen_braided_maze(nx, ny, seed)
    import random
    rng = random.Random(seed + 1)
    widths = {k: rng.choice([4.5, 6.0, cell, cell]) for k in conn}

    depot_cell = (nx // 2, ny // 2)
    corner_cells = {"bottom_left": (0, 0), "bottom_right": (nx - 1, 0),
                    "top_left": (0, ny - 1), "top_right": (nx - 1, ny - 1)}
    route_cells = {name: _maze_route(conn, depot_cell, cc)
                   for name, cc in corner_cells.items()}
    # Narrow the doorways along the I and L routes to ~1.5x the O-square (2 tiles),
    # so those blocks must turn their narrow side to squeeze through.
    narrow = 1.5 * 2 * tile_edge
    for name in ("bottom_right", "bottom_left"):       # I -> BR, L -> BL
        rc = route_cells[name]
        for ca, cb in zip(rc, rc[1:]):
            widths[frozenset((ca, cb))] = narrow

    x0, y0 = -nx * cell / 2.0, -ny * cell / 2.0
    center = lambda c: (x0 + (c[0] + 0.5) * cell, y0 + (c[1] + 0.5) * cell)
    walls = []

    def vwall(name, x, ylo, yhi):
        walls.append(_wall(name, (x, (ylo + yhi) / 2, wall_h),
                           (thick, (yhi - ylo) / 2, wall_h), solid=True))

    def hwall(name, y, xlo, xhi):
        walls.append(_wall(name, ((xlo + xhi) / 2, y, wall_h),
                           ((xhi - xlo) / 2, thick, wall_h), solid=True))

    vwall("b_w", x0, y0, y0 + ny * cell)
    vwall("b_e", x0 + nx * cell, y0, y0 + ny * cell)
    hwall("b_s", y0, x0, x0 + nx * cell)
    hwall("b_n", y0 + ny * cell, x0, x0 + nx * cell)
    for i in range(nx):
        for j in range(ny):
            c = (i, j)
            if i + 1 < nx:
                x = x0 + (i + 1) * cell
                ylo, yhi = y0 + j * cell, y0 + (j + 1) * cell
                k = frozenset((c, (i + 1, j)))
                if k not in conn:
                    vwall(f"v{i}_{j}", x, ylo, yhi)
                elif widths[k] < cell:
                    g = widths[k]; ym = (ylo + yhi) / 2
                    vwall(f"v{i}_{j}a", x, ylo, ym - g / 2)
                    vwall(f"v{i}_{j}b", x, ym + g / 2, yhi)
            if j + 1 < ny:
                y = y0 + (j + 1) * cell
                xlo, xhi = x0 + i * cell, x0 + (i + 1) * cell
                k = frozenset((c, (i, j + 1)))
                if k not in conn:
                    hwall(f"h{i}_{j}", y, xlo, xhi)
                elif widths[k] < cell:
                    g = widths[k]; xm = (xlo + xhi) / 2
                    hwall(f"h{i}_{j}a", y, xlo, xm - g / 2)
                    hwall(f"h{i}_{j}b", y, xm + g / 2, xhi)

    routes = {name: [center(c) for c in rc] for name, rc in route_cells.items()}

    th, hz = tile_edge / 2, thickness / 2
    blocks = ["Z", "I", "O", "L"]
    colors = {"Z": "0.93 0.11 0.14 1", "I": "0.0 0.80 0.95 1",   # Tetris colours:
              "O": "0.98 0.88 0.05 1", "L": "0.95 0.40 0.0 1"}   # Z red, I cyan, O yellow, L deep orange
    cx, cy = center(depot_cell)
    # offsets small enough that the longest block (I, 3.2 m) fits inside the
    # centre cell without poking through its walls
    depot = {"Z": (cx - 1.7, cy - 1.7), "I": (cx + 1.7, cy - 1.7),
             "O": (cx - 1.7, cy + 1.7), "L": (cx + 1.7, cy + 1.7)}
    deliver = {"Z": "top_left", "I": "bottom_right", "O": "top_right", "L": "bottom_left"}

    drones = "\n".join(_drone_body(i, (depot["Z"][0] + cell_offsets("Z", th)[i][0],
                                       depot["Z"][1] + cell_offsets("Z", th)[i][1],
                                       hz * 2 + tether_len),
                                   model=drone_model, tether=(0, 0, -tether_len),
                                   collide=False) for i in range(4))
    block_xml = "\n".join(_free_block_body(s, s, (*depot[s], hz), th, hz, colors[s])
                          for s in blocks)
    sites = "\n".join(
        f'    <site name="goal_{name}" pos="{center(cc)[0]} {center(cc)[1]} 0.05" '
        f'size="0.5" rgba="0.2 0.9 0.3 0.35"/>' for name, cc in corner_cells.items())
    assets = _x2_asset_block() if drone_model == "x2" else ""

    xml = f'''<mujoco model="tetraswarm_maze_relay">
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
{assets}
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse="0 0 0" ambient="0.65 0.65 0.65" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light name="top" dir="0 0 -1" directional="true" diffuse="0.55 0.55 0.55"/>
    <geom name="ground" type="plane" size="40 40 0.1" rgba="0.08 0.09 0.11 1"/>
{sites}
{chr(10).join(walls)}
{drones}
{block_xml}
  </worldbody>
</mujoco>'''

    info = {"blocks": blocks, "depot": depot, "deliver": deliver,
            "routes": routes, "depot_center": center(depot_cell),
            "grips": {s: cell_offsets(s, th) for s in blocks},
            "tile_half": th, "half_z": hz, "tether_len": tether_len,
            "nx": nx, "ny": ny, "cell": cell}
    return xml, info


if __name__ == "__main__":
    xml = build_scene(n_drones=10, payload_shape="L")
    print(xml)
