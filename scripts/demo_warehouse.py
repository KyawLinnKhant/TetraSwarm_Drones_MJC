"""
TetraSwarm warehouse mission — autonomous, in an UNKNOWN building.

A fleet of 4 lidar+camera drones starts at a dock (bottom-left) where 5 Tetris-coloured
tetromino blocks (I, L, O, Z, T) are packed. The drones know NOTHING about the maze.

  1. MAP the unknown maze by reactive WALL-FOLLOWING with CAMERA+LIDAR FUSION: each
     drone fuses a forward depth camera (dense front) with the lidar (sides + the
     occupancy map), 2 drones follow the left wall and 2 the right; they turn at
     corners (front+left -> right, front+right -> left), never back off, and share an
     occupancy grid until the room is mapped (~98%).
  2. Pick the DESTINATION = the GEOMETRIC FAR CORNER reachable from the dock (the
     mapped-free cell with the largest straight-line distance from the dock).
  3. TRANSPORT the 5 blocks ONE-BY-ONE with all 4 drones. Each block is slung under a
     compact square and carried along an A* route on the DISCOVERED map: the route
     stays inside mapped corridors, so the swarm goes AROUND walls (the solid block
     records 0 wall contacts — it never ghosts through). Blocks are set down packed
     side-by-side like Tetris at the far corner.

    python scripts/demo_warehouse.py               # headless, prints progress
    python scripts/demo_warehouse.py --gif         # + results/mission.gif + figures

Maze format follows robotics.farama.org/envs/maze/point_maze.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import heapq

import numpy as np
import mujoco
from scipy.optimize import linear_sum_assignment

from envs.scene_builder import build_pointmaze_scene, maze_from_grid
from scripts.demo_scout import OccupancyGrid, astar, find_frontiers
from scripts.demo_maze import arc_tools

DOCK_BLOCKS = ["I", "L", "O", "Z", "T"]


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--explore-seconds", dest="explore_seconds", type=float, default=300.0)
    ap.add_argument("--mappers", type=int, default=0,
                    help="number of mapping drones (0 = auto from dock lanes)")
    args = ap.parse_args()

    cell = 6.5
    mz = maze_from_grid(nx=6, ny=5, seed=args.seed, mission=True)   # sized so it maps fully
    xml, info = build_pointmaze_scene(mz, cell=cell, n_drones=4,
                                      dock_blocks=DOCK_BLOCKS, block_tile=0.15)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nd, nr = info["n_drones"], info["n_rays"]
    angles = np.array(info["angles"])
    x0, y0 = info["x0"], info["y0"]
    Wm, Hm = info["ncols"] * cell, info["nrows"] * cell
    max_range = 9.0
    sz = info["scout_z"]
    dock = np.array(info["center"])
    # All drones map (default); they DIVERGE at junctions toward unexplored branches
    # (the shared map tells them which corridors a pioneer already covered). --mappers
    # can hold a few back at the dock as base if you want.
    n_mappers = int(np.clip(args.mappers or nd, 1, nd))
    base_idxs = list(range(n_mappers, nd))
    print(f"deploying {n_mappers} mapper(s)"
          + (f" + {len(base_idxs)} base" if base_idxs else "") + "\n")

    # ---- ids: drones are force-driven solid boxes (mapping AND physical carry)
    sb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"scout{d}") for d in range(nd)]
    sdof = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"scout{d}_free")]
            for d in range(nd)]
    sqadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"scout{d}_free")]
             for d in range(nd)]
    sgeom = {d: {g for g in range(model.ngeom) if model.geom_bodyid[g] == sb[d]} for d in range(nd)}

    # ---- ids: blocks (solid free bodies, packed at the dock)
    shapes = info["dock_blocks"]
    bqadr = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_free")]
             for s in shapes}
    bvadr = {s: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_free")]
             for s in shapes}
    bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s) for s in shapes}

    ground_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    bgeoms = {s: {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[s]} for s in shapes}
    wall_hits = {s: 0 for s in shapes}

    def count_hits(carry):
        if carry is None:
            return
        for ci in range(data.ncon):
            c = data.contact[ci]
            in1, in2 = c.geom1 in bgeoms[carry], c.geom2 in bgeoms[carry]
            if in1 ^ in2:
                other = c.geom2 if in1 else c.geom1
                if other != ground_g and not any(other in sgeom[d] for d in range(nd)):
                    wall_hits[carry] += 1

    block_hz = info["block_hz"]
    GROUND_Z = block_hz                      # block resting on the floor
    CARRY_Z = 0.6                            # block centre while being carried
    TETHER = 0.6
    DRONE_MASS = 0.3
    drone_above = lambda cz: cz + block_hz + TETHER

    spawn_yaw = {s: info["blocks"][s]["yaw"] for s in shapes}
    pose = {s: (info["blocks"][s]["pos"][0], info["blocks"][s]["pos"][1],
                GROUND_Z, spawn_yaw[s]) for s in shapes}

    def write_blocks():
        for s in shapes:
            x, y, z, yaw = pose[s]
            data.qpos[bqadr[s]:bqadr[s] + 3] = [x, y, z]
            data.qpos[bqadr[s] + 3:bqadr[s] + 7] = yaw_quat(yaw)
            data.qvel[bvadr[s]:bvadr[s] + 6] = 0

    write_blocks()
    mujoco.mj_forward(model, data)

    # ---- shared occupancy map (built during exploration, used for routing)
    grid = OccupancyGrid(x0 - cell, y0 - cell, Wm + 2 * cell, Hm + 2 * cell, res=0.5)
    iw = lambda i, j: (grid.x0 + (i + 0.5) * grid.res, grid.y0 + (j + 0.5) * grid.res)

    goal_xy = [np.array(g) for g in info["goals"]]
    gsid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"goal{k}")
            for k in range(len(goal_xy))]

    def sense():
        rng = data.sensordata[:nd * nr].copy()
        for d in range(nd):
            sx, sy = float(data.xpos[sb[d]][0]), float(data.xpos[sb[d]][1])
            for i in range(nr):
                r = rng[d * nr + i]
                hit = 0 < r <= max_range
                r = r if hit else max_range
                grid.ray(sx, sy, sx + r * np.cos(angles[i]),
                         sy + r * np.sin(angles[i]), hit, drone=d)

    fg = info["free_grid"]
    free_world = [(x0 + (j + 0.5) * cell, y0 + (info["nrows"] - 1 - i + 0.5) * cell)
                  for i in range(fg.shape[0]) for j in range(fg.shape[1]) if fg[i, j]]
    free_ij = [grid.ij(wx, wy) for wx, wy in free_world]
    maze_cov = lambda: float(np.mean([grid.L[gi, gj] < -0.5 for gi, gj in free_ij]))

    # routes keep >=1.5 m clearance (3 grid cells) from walls so the carried block
    # (small, slung under a compact square) never clips a corner.
    TRANSPORT_INFL = 3

    def route_world(a, b, infl=TRANSPORT_INFL):
        """A* a polyline a->b on the DISCOVERED grid (mapped corridors only, with
        clearance for the block). None if unrouteable — NEVER a straight line."""
        p = astar(grid.L, grid.ij(*a), grid.ij(*b), infl=infl)
        if not p:
            p = astar(grid.L, grid.ij(*a), grid.ij(*b), infl=max(infl - 1, 1))
        if not p:
            return None
        pts = [iw(i, j) for i, j in p]
        pts[0], pts[-1] = tuple(a), tuple(b)
        return pts

    # ---- rendering (scene | map side-by-side) with a mission HUD
    frames = []
    renderer = cam = opt = None
    if args.gif:
        from PIL import Image, ImageDraw
        renderer = mujoco.Renderer(model, height=560, width=560)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0]
        cam.distance = max(Wm, Hm) * 1.18
        cam.azimuth, cam.elevation = 90, -89
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 1
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    def map_px(wx, wy):                              # world -> pixel on the map panel
        i, j = grid.ij(wx, wy)
        return (560 + i * 560.0 / grid.nx, (grid.ny - 1 - j) * 560.0 / grid.ny)

    def grab(status="", mark=None):
        if renderer is None:
            return
        from PIL import Image, ImageDraw
        renderer.update_scene(data, camera=cam, scene_option=opt)
        scene = Image.fromarray(renderer.render())
        mp = Image.fromarray(grid.image()).resize((560, 560), Image.NEAREST).convert("RGB")
        combo = Image.new("RGB", (1120, 560))
        combo.paste(scene, (0, 0))
        combo.paste(mp, (560, 0))
        dr = ImageDraw.Draw(combo)
        if mark is not None:                         # flash the destination on the map
            px, py = map_px(*mark)
            dr.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(40, 230, 90), width=3)
        if status:
            dr.rectangle([0, 0, 560, 22], fill=(12, 14, 18))
            dr.text((8, 5), status, fill=(235, 235, 240))
        frames.append(combo)

    # ============================================================ PHASE 1: MAP
    # CAMERA + LIDAR FUSION reactive WALL-FOLLOWING. The lidar (long 9 m range) builds the
    # map and reports left/right wall proximity; a forward DEPTH CAMERA per drone gives a
    # dense FRONT reading (fused: front = min(camera, lidar front cone)). Decisions use a
    # SHORT turn range. The drones keep following the wall and only TURN -- never back off:
    #   front+left blocked -> turn right;  front+right blocked -> turn left.
    dt0 = model.opt.timestep
    every = int(round((1 / 12) / dt0))
    sense_iv = max(1, int(0.1 / dt0))

    # forward depth camera: one small depth renderer, re-aimed per drone (a free viewpoint
    # sitting AT the drone looking along its heading) -> the camera half of the fusion.
    depth_r = mujoco.Renderer(model, 36, 48)
    depth_r.enable_depth_rendering()
    cam_d = mujoco.MjvCamera()
    cam_d.type = mujoco.mjtCamera.mjCAMERA_FREE
    depth_opt = mujoco.MjvOption()                   # hide the drones so each camera
    depth_opt.geomgroup[2] = 0                       # only sees WALLS (not other drones,
    depth_opt.geomgroup[3] = 0                       # which start clustered at the dock)
    cam_iv = max(1, int(0.12 / dt0))                 # ~8 Hz camera updates (staggered/drone)
    cam_front = {d: max_range for d in range(nd)}

    def cam_look(d, psi):
        # camera sits ~0.6 m AHEAD of the drone (clear of its own body) looking forward
        p = data.xpos[sb[d]]
        fx, fy = np.cos(psi), np.sin(psi)
        cam_d.lookat[:] = [p[0] + 0.9 * fx, p[1] + 0.9 * fy, sz]
        cam_d.distance, cam_d.azimuth, cam_d.elevation = 0.3, np.degrees(psi), 0.0
        depth_r.update_scene(data, camera=cam_d, scene_option=depth_opt)
        dep = depth_r.render()
        ch, cw = dep.shape
        # MEDIAN of a tight central patch (robust to the macOS depth near-clip artifact
        # that corrupts min()); +0.6 m brings it from the camera back to the drone frame.
        c = dep[2 * ch // 5:3 * ch // 5, 3 * cw // 8:5 * cw // 8]
        return min(float(np.median(c)) + 0.6, max_range)

    def cone_min(rngd, ctr, half):                   # min lidar range within +-half of ctr
        best = max_range
        for i in range(nr):
            if abs((angles[i] - ctr + np.pi) % (2 * np.pi) - np.pi) <= half:
                r = rngd[i]
                best = min(best, r if 0 < r <= max_range else max_range)
        return best

    R_TURN = 0.5 * cell                              # SHORT decision range (~3.2 m)
    WALL_TGT = 0.40 * cell                           # desired distance to the followed wall
    KV, VCRUISE, FMAX_E, TURN = 3.0, 3.5, 24.0, 0.008    # velocity gain, cruise m/s, turn/step
    cov_t, best_cov, stall = [], 0.0, 0
    sense()
    hand = {d: (1.0 if d % 2 == 0 else -1.0) for d in range(nd)}   # +1 left-hand, -1 right
    head = {}
    for d in range(nd):                              # face the most-open direction to start
        rngd = data.sensordata[d * nr:(d + 1) * nr]
        head[d] = max(angles, key=lambda a: cone_min(rngd, a, 0.2))
    last_pos2 = {d: np.array(data.xpos[sb[d]][:2]) for d in range(nd)}
    stuck2 = {d: 0.0 for d in range(nd)}
    chk_iv = max(1, int(0.5 / dt0))
    print(f"PHASE 1: wall-following ({nd} drones, camera+lidar fusion, no back-off) ...")
    for k in range(int(args.explore_seconds / dt0)):
        if k % sense_iv == 0:
            sense()
        for d in range(nd):
            p = data.xpos[sb[d]]
            rngd = data.sensordata[d * nr:(d + 1) * nr]
            th, h = head[d], hand[d]
            if k % cam_iv == d:                      # staggered camera update for drone d
                cam_front[d] = cam_look(d, th)
            # FUSION: dense camera front + lidar front cone; lidar for the sides
            front = min(cam_front[d], cone_min(rngd, th, 0.35))
            left = cone_min(rngd, th + np.pi / 2, 0.5)
            right = cone_min(rngd, th - np.pi / 2, 0.5)
            fb, lb, rb = front < R_TURN, left < R_TURN, right < R_TURN
            dL = left if h > 0 else right             # distance to the followed wall
            # Heading turns ONLY at corners (held steady in straight corridors -> NO spin):
            if fb:                                    # obstacle ahead -> turn per the rules
                if lb and not rb:
                    dpsi = -TURN; vf = 0.3            # front + left  -> turn right
                elif rb and not lb:
                    dpsi = +TURN; vf = 0.3            # front + right -> turn left
                elif lb and rb:                       # DEAD-END: pivot IN PLACE (no forward
                    dpsi = TURN if left > right else -TURN   # ram) + lateral slides off the
                    vf = 0.0                          # wall, so it rotates out instead of stuck
                else:
                    dpsi = h * TURN; vf = 0.3         # open T-junction -> turn toward the hand
            else:                                     # front open -> follow the wall
                # forward-DIAGONAL sensor looks ahead along the followed wall, so we round
                # an outside corner only until it re-acquires the wall -> no circling.
                d_diag = cone_min(rngd, th + h * np.pi / 4, 0.3)
                if dL > 1.7 * WALL_TGT and d_diag > 1.7 * WALL_TGT:
                    dpsi = h * 0.6 * TURN             # outside corner: curve toward the wall...
                else:
                    dpsi = 0.0                        # ...wall back in view -> hold heading
                vf = 1.0                              # (lateral force keeps the hug distance)
            head[d] = th + np.clip(dpsi, -TURN, TURN)
            th = head[d]
            # HOLONOMIC velocity command: cruise forward along heading + lateral wall-hug
            fwd = np.array([np.cos(th), np.sin(th)])
            lat = np.array([np.cos(th + h * np.pi / 2), np.sin(th + h * np.pi / 2)])  # toward wall
            latv = np.clip(0.9 * (dL - WALL_TGT), -1.5, 1.5) if dL < 1.7 * WALL_TGT else 0.0
            vdes = VCRUISE * vf * fwd + latv * lat
            vel = data.qvel[sdof[d]:sdof[d] + 3]
            f = np.zeros(3)
            f[:2] = KV * (vdes - vel[:2])
            for e in range(nd):                       # light separation only (no wall push-back)
                if e != d:
                    dv = p[:2] - data.xpos[sb[e]][:2]; dn = np.hypot(*dv)
                    if 1e-3 < dn < 1.6:
                        f[:2] += 8.0 * dv / dn
            f[2] = 8.0 * (sz - p[2]) - 5.0 * vel[2] + DRONE_MASS * 9.81   # hold altitude
            mag = np.linalg.norm(f)
            if mag > FMAX_E:
                f *= FMAX_E / mag
            data.xfrc_applied[sb[d], :3] = f
        write_blocks()
        mujoco.mj_step(model, data)
        for d in range(nd):                          # mark the drone's own cell free (trail)
            gi, gj = grid.ij(data.xpos[sb[d]][0], data.xpos[sb[d]][1])
            if 0 <= gi < grid.nx and 0 <= gj < grid.ny:
                grid.L[gi, gj] = min(grid.L[gi, gj], -2.0)
                if grid.owner[gi, gj] < 0:
                    grid.owner[gi, gj] = d
        t = k * dt0
        if k % chk_iv == 0:                          # recovery: a drone with no new ground
            front = find_frontiers(grid.L)           # nearby re-aims toward the nearest
            for d in range(nd):                      # UNEXPLORED edge, then wall-follows there
                cur = np.array(data.xpos[sb[d]][:2])
                if np.hypot(*(cur - last_pos2[d])) < 0.7:
                    stuck2[d] += chk_iv * dt0
                    if stuck2[d] >= 2.5:
                        if len(front):
                            dij = np.array(grid.ij(cur[0], cur[1]))
                            nf = front[np.argmin(np.linalg.norm(front - dij, axis=1))]
                            wx, wy = iw(nf[0], nf[1])
                            head[d] = np.arctan2(wy - cur[1], wx - cur[0])
                        else:
                            rngd = data.sensordata[d * nr:(d + 1) * nr]
                            head[d] = max(angles, key=lambda a: cone_min(rngd, a, 0.2))
                        stuck2[d] = 0.0
                else:
                    stuck2[d] = 0.0
                last_pos2[d] = cur
        if k % int(10 / dt0) == 0:
            print(f"  t={t:4.0f}s  cov={100 * maze_cov():3.0f}%")
        if k % int(2 / dt0) == 0:
            cov_now = maze_cov()
            cov_t.append((t, cov_now))
            if cov_now > best_cov + 0.003:
                best_cov, stall = cov_now, 0
            else:
                stall += 1
                if stall % 2 == 0:                   # ~4 s with no new ground -> the drones are
                    front = find_frontiers(grid.L)   # circulating; re-aim them at spread-out
                    if len(front):                   # frontiers and let them wall-follow there
                        taken = []
                        for d in range(nd):
                            cd = data.xpos[sb[d]][:2]
                            dij = np.array(grid.ij(cd[0], cd[1]))
                            dist = np.linalg.norm(front - dij, axis=1).astype(float)
                            for tk in taken:
                                dist += 300.0 * (np.linalg.norm(front - tk, axis=1) < 6)
                            nf = front[int(np.argmin(dist))]; taken.append(nf)
                            wx, wy = iw(nf[0], nf[1])
                            head[d] = np.arctan2(wy - cd[1], wx - cd[0])
        if args.gif and k % every == 0:
            grab(status=f"MAPPING  {100 * maze_cov():.0f}%   ({nd} drones, wall-following cam+lidar)")
        if maze_cov() > 0.97 or stall >= 30:         # wall-following loops -> stop at plateau
            break

    depth_r.close()
    cov = maze_cov()
    print(f"  -> mapping done: {100 * cov:.0f}% of corridors, t={k * dt0:.0f}s\n")
    data.xfrc_applied[:] = 0

    # ============================================================ PHASE 2: FAR CORNER
    # Destination = the GEOMETRIC far corner: of all mapped-free, still-reachable cells,
    # the one with the largest straight-line distance from the dock.
    best = None
    for (wx, wy) in free_world:
        if grid.L[grid.ij(wx, wy)] < -0.5 and route_world(dock, (wx, wy)) is not None:
            dd = float(np.hypot(wx - dock[0], wy - dock[1]))
            if best is None or dd > best[1]:
                best = ((wx, wy), dd)
    dest = np.array(best[0]) if best else np.array(goal_xy[0])
    print(f"PHASE 2: geometric far corner = ({dest[0]:.0f}, {dest[1]:.0f}), "
          f"{best[1]:.0f} m from dock\n")
    # flash a small marker at the corner once on the map, then hide it
    model.site_pos[gsid[0]] = [dest[0], dest[1], 0.15]
    model.site_rgba[gsid[0]] = [0.15, 0.95, 0.35, 0.95]
    mujoco.mj_forward(model, data)
    if args.gif:
        for _ in range(14):
            grab(status=f"FAR CORNER FOUND  ({dest[0]:.0f}, {dest[1]:.0f})", mark=dest)
    model.site_rgba[gsid[0]] = [0.15, 0.95, 0.35, 0.0]      # disappear
    mujoco.mj_forward(model, data)

    slot = {s: (dest[0] + (info["blocks"][s]["pos"][0] - dock[0]),
                dest[1] + (info["blocks"][s]["pos"][1] - dock[1])) for s in shapes}

    # ============================================================ PHASE 3: TRANSPORT
    # KINEMATIC carry: the drones + slung block snap to commanded targets that follow an
    # A* route computed on the DISCOVERED map. The route stays inside mapped corridors
    # (>=1.5 m clearance), so the swarm goes AROUND walls and the SOLID block records 0
    # wall contacts -- it never ghosts through a wall. The block hangs under a compact
    # square formation so it always fits the corridor.
    SQ = np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    dt = model.opt.timestep
    kc = [0]
    every_t = int(round((1 / 12) / dt)) if args.gif else 1
    hud = [""]

    def square(cx, cy, dz):
        return np.column_stack([np.array([cx, cy]) + SQ, np.full(len(SQ), dz)])

    def step(targets, carry=None, cz=CARRY_Z, block_yaw=0.0):
        for d in range(nd):
            a = sqadr[d]
            data.qpos[a:a + 3] = targets[d]
            data.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        if carry is not None:
            cen = targets[:, :2].mean(0)
            pose[carry] = (float(cen[0]), float(cen[1]), cz, block_yaw)
        write_blocks()
        mujoco.mj_forward(model, data)
        count_hits(carry)
        if args.gif and kc[0] % every_t == 0:
            grab(status=hud[0])
        kc[0] += 1

    def fly(route, carry, block_yaw, speed=4.0, cz=CARRY_Z):
        if len(route) < 2:
            return tuple(route[-1])
        path, _ = arc_tools(route)
        total = max(np.sum(np.linalg.norm(np.diff(np.array(route), axis=0), axis=1)), 1e-6)
        prog = 0.0
        cx, cy = path(0.0)
        while prog < 1.0:
            prog = min(1.0, prog + speed / total * dt)
            cx, cy = path(prog)
            step(square(cx, cy, drone_above(cz)), carry, cz, block_yaw)
        return (cx, cy)

    def vary_cz(cx, cy, carry, block_yaw, z0, z1, seconds):
        n = int(seconds / dt)
        for k in range(n):
            a = (k + 1) / n
            cz = z0 + a * (z1 - z0)
            step(square(cx, cy, drone_above(cz)), carry, cz, block_yaw)

    # regroup at the dock: route EACH drone home through mapped corridors (A*), never
    # a straight line -> no ghosting back to the dock.
    hud[0] = "REGROUP AT DOCK"
    home = []
    for d in range(nd):
        slotw = tuple(dock + SQ[d])
        r = route_world(tuple(data.qpos[sqadr[d]:sqadr[d] + 2]), slotw)
        home.append(arc_tools(r)[0] if r and len(r) >= 2 else None)
        if home[-1] is None:                          # already home / unrouteable
            home[-1] = (lambda a, q=slotw: np.array(q))
    n = int(4.0 / dt)
    for k in range(n):
        a = (k + 1) / n
        tg = np.array([[*home[d](a), drone_above(CARRY_Z)] for d in range(nd)])
        step(tg)

    print("PHASE 3: transporting blocks one-by-one (kinematic carry on mapped routes)\n")
    log, all_routes = [], []
    cur = tuple(dock)
    for bi, s in enumerate(shapes):
        dep = info["blocks"][s]["pos"]
        yaw = spawn_yaw[s]
        rw = route_world(dock, tuple(dest))
        if rw is None:
            print(f"  !! no mapped route dock->corner; cannot carry {s}")
            continue
        route = [tuple(dep)] + rw[1:] + [slot[s]]
        all_routes.append(route)
        # fly empty over the block, descend to grip, lift
        hud[0] = f"FETCH {s}   ({bi + 1}/{len(shapes)})"
        fly([cur, dep], None, yaw, speed=4.5, cz=CARRY_Z)
        vary_cz(dep[0], dep[1], None, yaw, CARRY_Z, GROUND_Z, 0.9)
        vary_cz(dep[0], dep[1], s, yaw, GROUND_Z, CARRY_Z, 1.0)
        # carry it through the mapped corridors to its Tetris slot
        hud[0] = f"CARRY {s}   ({bi + 1}/{len(shapes)})  ->  far corner"
        c = fly(route, s, yaw, speed=3.5, cz=CARRY_Z)
        vary_cz(c[0], c[1], s, yaw, CARRY_Z, GROUND_Z, 1.0)
        bx, by = data.xpos[bid[s]][:2]
        pose[s] = (float(bx), float(by), GROUND_Z, yaw)
        d_err = float(np.linalg.norm(np.array([bx, by]) - np.array(slot[s])))
        log.append(d_err)
        print(f"  delivered {s} -> slot ({slot[s][0]:.0f}, {slot[s][1]:.0f})  err {d_err:.2f} m")
        # fly empty back toward the dock for the next block
        if bi < len(shapes) - 1:
            hud[0] = "RETURN EMPTY"
            back = route_world(tuple(dest), dock)
            if back:
                fly(back, None, 0.0, speed=5.0, cz=CARRY_Z)
            cur = tuple(dock)
        else:
            cur = tuple(c)

    if args.gif:
        hud[0] = "MISSION COMPLETE"
        for _ in range(20):
            grab(status=hud[0])

    total_hits = sum(wall_hits.values())
    print(f"\nMISSION COMPLETE: {len(log)} blocks delivered, "
          f"max placement err {max(log):.2f} m")
    print(f"block<->wall contacts (routes stay in mapped corridors): {wall_hits} (total {total_hits})")
    print("PASS" if max(log) < 1.0 and total_hits == 0 else "FAIL")

    # ---- outputs: map + research figures
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    from PIL import Image
    Image.fromarray(grid.image()).resize((600, 600), Image.NEAREST).save(
        os.path.join(ROOT, "results", "mission_map.png"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    # (1) per-drone coverage: each mapper's free cells in its own colour
    dcols = np.array([[0.90, 0.15, 0.15], [0.20, 0.55, 0.95],
                      [0.20, 0.80, 0.40], [0.95, 0.75, 0.20]])
    rgb = np.ones((grid.ny, grid.nx, 3))
    occ, free, own = (grid.L > 0.5).T, (grid.L < -0.5).T, grid.owner.T
    for d in range(nd):
        rgb[free & (own == d)] = dcols[d % 4]
    rgb[occ] = (0.08, 0.08, 0.10)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb[::-1], interpolation="nearest")
    ax.set(title=f"Warehouse mapping — {n_mappers} scouts + {len(base_idxs)} base "
                 f"(coverage by drone)", xticks=[], yticks=[])
    cover = [int((free & (own == d)).sum()) for d in range(nd)]
    labels = [Patch(color=dcols[d % 4],
                    label=f"{'base' if d in base_idxs else 'scout ' + str(d)}: {cover[d]} cells")
              for d in range(nd)] + [Patch(color=(0.08, 0.08, 0.10), label="wall")]
    ax.legend(handles=labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "warehouse_coverage.png"), dpi=130)
    plt.close(fig)

    # (2) coverage vs time
    if cov_t:
        ts, cs = zip(*cov_t)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ts, 100 * np.array(cs), color="0.2", lw=2)
        ax.set(xlabel="time (s)", ylabel="maze mapped (%)",
               title=f"{n_mappers}-drone mapping coverage", ylim=(0, 101))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "results", "figures", "warehouse_coverage_time.png"), dpi=130)
        plt.close(fig)

    # (3) planned A* transport routes over the discovered map
    res = grid.res
    extent = [grid.x0, grid.x0 + grid.nx * res, grid.y0, grid.y0 + grid.ny * res]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid.image(), extent=extent, origin="lower", interpolation="nearest")
    rcols = ["#00cce8", "#f28200", "#f0d800", "#e62633", "#9b4ed8"]
    for ri, route in enumerate(all_routes):
        xs, ys = zip(*route)
        ax.plot(xs, ys, color=rcols[ri % len(rcols)], lw=2, label=shapes[ri])
    ax.scatter([dock[0]], [dock[1]], c="w", s=80, marker="s", label="dock")
    ax.scatter([dest[0]], [dest[1]], c="#2ee65a", s=120, marker="*", label="far corner")
    ax.set(title="Transport routes on the discovered map", xticks=[], yticks=[])
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "warehouse_routes.png"), dpi=130)
    plt.close(fig)
    print("wrote results/mission_map.png + results/figures/warehouse_{coverage,coverage_time,routes}.png")

    if args.gif and frames:
        renderer.close()
        fr = frames[::3]
        pal = fr[len(fr) // 2].quantize(colors=160)
        q = [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in fr]
        out = os.path.join(ROOT, "results", "mission.gif")
        q[0].save(out, save_all=True, append_images=q[1:], duration=90, loop=0, optimize=True)
        print(f"wrote {out}  ({len(q)} frames)")


if __name__ == "__main__":
    main()
