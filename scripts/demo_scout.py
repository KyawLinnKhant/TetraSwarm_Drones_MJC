"""
Multi-drone autonomous exploration of an UNKNOWN maze with onboard lidar
(no overhead camera, no prior map).

Each drone ray-casts a 24-beam lidar into a SHARED log-odds occupancy grid (free
along each beam, occupied at the hit, unknown otherwise). The team then runs
classical decentralized FRONTIER exploration: detect frontiers (free/unknown
boundary), cluster them, assign each drone a large reachable cluster (spread
apart), plan an A* path over the sensed free space, and drive there behind a
reactive lidar-proximity avoidance layer (so the solid drones never touch a wall
or each other). They split the maze and merge one map.

    python scripts/demo_scout.py --drones 4 --gif

NOTE ON NAMING: this is a *classical frontier-based* explorer (Yamauchi-style),
INSPIRED BY the informative-exploration goal of MarmotLab's VIPER. It is NOT the
learned VIPER policy; swapping the frontier heuristic for a learned
informative-path-planning policy (e.g. VIPER) is the natural drop-in and the
intended next step.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_scout_scene


import heapq


def find_frontiers(L):
    """Frontier cells = known-FREE grid cells with an UNKNOWN 4-neighbour — the
    edge of the explored region. Returns an (m,2) array of (i,j)."""
    free = L < -0.5
    unk = np.abs(L) <= 0.5
    f = np.zeros_like(free)
    f[1:-1, 1:-1] = free[1:-1, 1:-1] & (unk[2:, 1:-1] | unk[:-2, 1:-1] |
                                        unk[1:-1, 2:] | unk[1:-1, :-2])
    return np.argwhere(f)


def astar(L, start, goal, infl=3):
    """8-connected A* on the occupancy grid. Walk known-FREE cells; never enter
    unknown cells (could be wall). Only the WALLS are dilated by ``infl`` cells so
    the drone keeps clearance from them while still being able to drive up to the
    frontier (the free/unknown boundary). Returns a list of (i,j) or []."""
    occ = L > 0.5
    for _ in range(infl):                               # dilate walls only
        b = occ.copy()
        b[1:-1, 1:-1] |= (occ[2:, 1:-1] | occ[:-2, 1:-1] |
                          occ[1:-1, 2:] | occ[1:-1, :-2])
        occ = b
    blocked = occ | (np.abs(L) <= 0.5)                  # walls(+clearance) + unknown
    blocked[goal] = False
    nx, ny = L.shape
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    h = lambda a: np.hypot(a[0] - goal[0], a[1] - goal[1])
    openh = [(h(start), 0.0, start)]
    came, g = {start: None}, {start: 0.0}
    while openh:
        _, gc, c = heapq.heappop(openh)
        if c == goal:
            path = [c]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        for dx, dy in moves:
            ni, nj = c[0] + dx, c[1] + dy
            if 0 <= ni < nx and 0 <= nj < ny and not blocked[ni, nj]:
                ng = gc + np.hypot(dx, dy)
                if ng < g.get((ni, nj), 1e18):
                    g[(ni, nj)] = ng
                    came[(ni, nj)] = c
                    heapq.heappush(openh, (ng + h((ni, nj)), ng, (ni, nj)))
    return []


class OccupancyGrid:
    """Log-odds 2D occupancy grid; also records WHICH drone first mapped each cell
    (coverage attribution) for the per-drone graph."""
    def __init__(self, x0, y0, w, h, res=0.4):
        self.x0, self.y0, self.res = x0, y0, res
        self.nx, self.ny = int(w / res) + 1, int(h / res) + 1
        self.L = np.zeros((self.nx, self.ny))      # log-odds, 0 = unknown
        self.owner = np.full((self.nx, self.ny), -1, int)   # drone that mapped it

    def ij(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def ray(self, x0, y0, x1, y1, hit, drone=0):
        i0, j0 = self.ij(x0, y0)
        i1, j1 = self.ij(x1, y1)
        n = max(abs(i1 - i0), abs(j1 - j0), 1)
        for t in range(n + 1):
            i = int(round(i0 + (i1 - i0) * t / n))
            j = int(round(j0 + (j1 - j0) * t / n))
            if 0 <= i < self.nx and 0 <= j < self.ny:
                self.L[i, j] = np.clip(self.L[i, j] - 0.4, -6, 6)   # free
                if self.owner[i, j] < 0:
                    self.owner[i, j] = drone
        if hit and 0 <= i1 < self.nx and 0 <= j1 < self.ny:
            self.L[i1, j1] = np.clip(self.L[i1, j1] + 0.85, -6, 6)  # occupied
            if self.owner[i1, j1] < 0:
                self.owner[i1, j1] = drone

    def image(self):
        """RGB: unknown=grey, free=dark, occupied=white."""
        img = np.full((self.ny, self.nx, 3), 90, np.uint8)         # unknown
        free = self.L < -0.5
        occ = self.L > 0.5
        img[free.T] = (28, 30, 36)
        img[occ.T] = (235, 235, 240)
        return img[::-1]                                           # y up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--drones", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=40.0)
    args = ap.parse_args()

    xml, info = build_scout_scene(n_drones=args.drones)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nr, nd = info["n_rays"], info["n_drones"]
    angles = np.array(info["angles"])
    max_range = 7.0      # short-range ToF sensing horizon

    cell, x0, y0 = info["cell"], info["x0"], info["y0"]
    ctr = info["center"]
    grid = OccupancyGrid(x0 - cell, y0 - cell,
                         (info["nx"] + 2) * cell, (info["ny"] + 2) * cell, res=0.4)
    res = grid.res
    iw = lambda i, j: (grid.x0 + (i + 0.5) * res, grid.y0 + (j + 0.5) * res)
    sb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"scout{d}") for d in range(nd)]
    sdof = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"scout{d}_free")]
            for d in range(nd)]
    sz = info["scout_z"]
    bg = {d: {g for g in range(model.ngeom)
              if model.geom_bodyid[g] == sb[d]} for d in range(nd)}
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    def sense():
        rng = data.sensordata[:nd * nr].copy()
        for d in range(nd):
            sx, sy = float(data.xpos[sb[d]][0]), float(data.xpos[sb[d]][1])
            for i in range(nr):
                r = rng[d * nr + i]
                # cap at the sensing horizon: beyond it, treat as free-up-to-horizon
                # (no hit) so the drone must physically VISIT regions to map them.
                if r < 0 or r > max_range:
                    hit, r = False, max_range
                else:
                    hit = True
                grid.ray(sx, sy, sx + r * np.cos(angles[i]),
                         sy + r * np.sin(angles[i]), hit, drone=d)

    gpath = [[] for _ in range(nd)]
    gidx = [0] * nd

    ccell = np.array(grid.ij(*ctr))

    def replan():
        """Cluster the frontiers, then DISPERSE the drones by angular sector — each
        drone takes the largest reachable cluster in its own slice of the maze, so
        the team fans out to different regions and explores in parallel (the
        multi-drone speedup). Returns False when no frontiers remain."""
        front = find_frontiers(grid.L)
        if len(front) == 0:
            return False
        keys = (front[:, 0] // 6) * 100000 + (front[:, 1] // 6)
        uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
        reps, sizes = [], []
        for ci in range(len(uk)):
            pts = front[inv == ci]
            c = pts.mean(0)
            reps.append(pts[np.argmin(np.linalg.norm(pts - c, axis=1))])
            sizes.append(cnt[ci])
        reps, sizes = np.array(reps), np.array(sizes, float)
        ang = np.arctan2(reps[:, 1] - ccell[1], reps[:, 0] - ccell[0]) % (2 * np.pi)
        for d in range(nd):
            dij = np.array(grid.ij(*data.xpos[sb[d]][:2]))
            lo, hi = d * 2 * np.pi / nd, (d + 1) * 2 * np.pi / nd
            in_sec = np.where((ang >= lo) & (ang < hi))[0] if nd > 1 \
                else np.arange(len(reps))
            cand = in_sec if len(in_sec) else np.arange(len(reps))
            # prefer large clusters; tie-break by nearness to the drone
            score = sizes[cand] - 0.3 * np.linalg.norm(reps[cand] - dij, axis=1)
            chosen = None
            for idx in cand[np.argsort(-score)]:
                p = astar(grid.L, tuple(dij), tuple(reps[idx]))
                if p:
                    chosen = p
                    break
            gpath[d] = chosen[:max(1, len(chosen) - 3)] if chosen else []
            gidx[d] = 0
        return True

    frames = []
    if args.gif:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=560, width=560)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0]
        cam.distance = max(info["nx"], info["ny"]) * cell * 1.35
        cam.azimuth, cam.elevation = 90, -89
    every = int(round((1 / 12) / model.opt.timestep))

    # maze-interior region of the grid (the only mappable part; the border can't
    # be sensed) — coverage is reported over THIS, not the whole padded grid.
    mlo, mhi = int(cell / res), grid.nx - int(cell / res)
    maze_cov = lambda: float(np.mean(np.abs(grid.L[mlo:mhi, mlo:mhi]) > 0.5))

    steps = int(args.seconds / model.opt.timestep)
    plan_every = int(0.5 / model.opt.timestep)
    sense()
    explored, wall_hits, dd_hits = True, 0, 0
    cov_t = []                                          # (time, maze coverage)
    for k in range(steps):
        if k % 8 == 0:
            sense()
        if k % plan_every == 0:
            explored = replan()
        for d in range(nd):
            p = data.xpos[sb[d]]
            gp = gpath[d]
            if gp:
                while gidx[d] < len(gp) - 1 and \
                        np.hypot(*(np.array(iw(*gp[gidx[d]])) - p[:2])) < 0.7:
                    gidx[d] += 1
                wx, wy = iw(*gp[min(gidx[d], len(gp) - 1)])
            else:
                wx, wy = ctr                           # no frontier -> retreat to centre
            f = 6.5 * (np.array([wx, wy, sz]) - p) - 5.0 * data.qvel[sdof[d]:sdof[d] + 3]
            # REACTIVE OBSTACLE AVOIDANCE (highest-priority layer): the lidar is a
            # proximity ring; any beam closer than SAFE pushes the drone directly
            # away, so it never grinds a wall even if the planned path cut a corner.
            SAFE = 1.05
            rngd = data.sensordata[d * nr:(d + 1) * nr]
            for i in range(nr):
                r = rngd[i]
                if 0 < r < SAFE:
                    push = 16.0 * (SAFE - r) / SAFE
                    f[0] -= push * np.cos(angles[i])
                    f[1] -= push * np.sin(angles[i])
            for e in range(nd):                        # separation (avoid each other)
                if e != d:
                    dv = p[:2] - data.xpos[sb[e]][:2]
                    dn = np.hypot(*dv)
                    if 1e-3 < dn < 1.6:
                        f[:2] += 14.0 * dv / dn
            f[2] += 0.3 * 9.81
            m = np.linalg.norm(f)
            if m > 18.0:
                f *= 18.0 / m
            data.xfrc_applied[sb[d], :3] = f
        mujoco.mj_step(model, data)
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            in_d = [d for d in range(nd) if g1 in bg[d] or g2 in bg[d]]
            if in_d:
                other = g2 if g1 in bg[in_d[0]] else g1
                if other == ground:
                    continue
                if any(other in bg[e] for e in range(nd)):
                    dd_hits += 1
                else:
                    wall_hits += 1
        if args.gif and k % every == 0:
            renderer.update_scene(data, camera=cam)
            scene = Image.fromarray(renderer.render())
            mp = Image.fromarray(grid.image()).resize((560, 560), Image.NEAREST).convert("RGB")
            combo = Image.new("RGB", (1120, 560))
            combo.paste(scene, (0, 0))
            combo.paste(mp, (560, 0))
            frames.append(combo)
        if k % int(1.0 / model.opt.timestep) == 0:
            cov_t.append((data.time, maze_cov()))
        if not explored and k > plan_every:
            break                                      # map complete -> done

    known = maze_cov()
    print(f"{nd} drones explored autonomously (frontier-based, no map given): "
          f"{100 * known:.0f}% of the MAZE mapped in {data.time:.0f}s")
    print(f"collisions during exploration -> walls: {wall_hits}, drone-drone: {dd_hits}")

    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    np.save(os.path.join(ROOT, "results", f"cov_{nd}drones.npy"), np.array(cov_t))
    from PIL import Image
    Image.fromarray(grid.image()).resize((600, 600), Image.NEAREST).save(
        os.path.join(ROOT, "results", "scout_map.png"))
    print("wrote results/scout_map.png")

    # per-drone coverage graph: each drone's mapped free space in its own colour,
    # walls in black, unknown white.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    dcols = np.array([[0.90, 0.15, 0.15], [0.20, 0.55, 0.95],
                      [0.20, 0.80, 0.40], [0.95, 0.75, 0.20]])
    rgb = np.ones((grid.ny, grid.nx, 3))                        # unknown = white
    occ = (grid.L > 0.5).T
    free = (grid.L < -0.5).T
    own = grid.owner.T
    for d in range(nd):
        rgb[free & (own == d)] = dcols[d % 4]
    rgb[occ] = (0.08, 0.08, 0.10)                               # walls black
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb[::-1], interpolation="nearest")
    ax.set(title=f"Cooperative frontier exploration — {nd} drones (coverage by drone)",
           xticks=[], yticks=[])
    cover = [int((free & (own == d)).sum()) for d in range(nd)]
    ax.legend(handles=[Patch(color=dcols[d % 4], label=f"drone {d}: {cover[d]} cells")
                       for d in range(nd)] + [Patch(color=(0.08, 0.08, 0.10), label="wall")],
              loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "scout_coverage.png"), dpi=130)
    plt.close(fig)
    print("wrote results/figures/scout_coverage.png  (per-drone coverage)")
    if args.gif and frames:
        frames = frames[::2]                                       # ~6 fps
        frames = [f.resize((840, 420), Image.BILINEAR) for f in frames]
        pal = frames[len(frames) // 2].quantize(colors=128)
        q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
        q[0].save(os.path.join(ROOT, "results", "scout.gif"), save_all=True,
                  append_images=q[1:], duration=120, loop=0, optimize=True)
        print(f"wrote results/scout.gif ({len(q)} frames)")


if __name__ == "__main__":
    main()
