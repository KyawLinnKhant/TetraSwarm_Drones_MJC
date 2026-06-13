"""
Unknown PointMaze exploration: the drones know NOTHING — not the maze, not where
the drop zones are. They fan out with onboard lidar, build a shared occupancy map
of an unknown Farama-PointMaze-format maze, and DISCOVER the goal / drop-zone cells
(they light up when a drone senses one). Frontier-based exploration + a reactive
lidar-proximity layer (solid drones, no wall/drone collisions).

    python scripts/demo_pointmaze.py --gif        # results/pointmaze.gif + map + coverage

Maze format follows robotics.farama.org/envs/maze/point_maze (1=wall,0=free,'g'=goal,'r'=start).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_pointmaze_scene, maze_from_grid
from scripts.demo_scout import OccupancyGrid, find_frontiers, astar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=70.0)
    args = ap.parse_args()

    mz = maze_from_grid(nx=6, ny=5, seed=args.seed, n_goals=4)
    xml, info = build_pointmaze_scene(mz, cell=6.5, n_drones=4)   # 4-drone fleet (same as relay)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nd, nr = info["n_drones"], info["n_rays"]
    angles = np.array(info["angles"])
    cell, x0, y0 = info["cell"], info["x0"], info["y0"]
    Wm, Hm = info["ncols"] * cell, info["nrows"] * cell
    max_range = 9.0                                  # moderate -> must traverse to map

    grid = OccupancyGrid(x0 - cell, y0 - cell, Wm + 2 * cell, Hm + 2 * cell, res=0.5)
    res = grid.res
    iw = lambda i, j: (grid.x0 + (i + 0.5) * res, grid.y0 + (j + 0.5) * res)
    sb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"scout{d}") for d in range(nd)]
    sdof = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"scout{d}_free")]
            for d in range(nd)]
    bg = {d: {g for g in range(model.ngeom) if model.geom_bodyid[g] == sb[d]} for d in range(nd)}
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    sz = info["scout_z"]

    # goal/drop-zone sites: hidden until a drone discovers them
    gsid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"goal{k}")
            for k in range(len(info["goals"]))]
    goal_xy = [np.array(g) for g in info["goals"]]
    found = set()

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

    def discover():
        for k, gx in enumerate(goal_xy):
            if k not in found:
                dmin = min(np.hypot(*(data.xpos[sb[d]][:2] - gx)) for d in range(nd))
                if dmin < 1.6 * cell:                 # a drone got close enough to see it
                    found.add(k)
                    model.site_rgba[gsid[k]] = [0.15, 0.95, 0.35, 0.95]
                    print(f"  discovered drop zone {k} at ({gx[0]:.0f}, {gx[1]:.0f})")

    gpath = [[] for _ in range(nd)]
    gidx = [0] * nd
    ccell = np.array(grid.ij(*info["center"]))

    def replan():
        front = find_frontiers(grid.L)
        if len(front) == 0:
            return False
        keys = (front[:, 0] // 6) * 100000 + (front[:, 1] // 6)
        uk, inv, _ = np.unique(keys, return_inverse=True, return_counts=True)
        reps = np.array([front[inv == ci][np.argmin(np.linalg.norm(
            front[inv == ci] - front[inv == ci].mean(0), axis=1))] for ci in range(len(uk))])
        taken = []
        for d in range(nd):                          # each drone -> nearest REACHABLE
            dij = np.array(grid.ij(*data.xpos[sb[d]][:2]))  # frontier, spread from others
            dist = np.linalg.norm(reps - dij, axis=1).astype(float)
            for t in taken:
                dist += 400.0 * (np.linalg.norm(reps - t, axis=1) < 10)
            gpath[d] = []
            for idx in np.argsort(dist):
                p = astar(grid.L, tuple(dij), tuple(reps[idx]), infl=2)
                if p:
                    gpath[d] = p
                    gidx[d] = 0
                    taken.append(reps[idx])
                    break
        return True

    frames = []
    if args.gif:
        from PIL import Image
        renderer = mujoco.Renderer(model, height=560, width=560)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0]
        cam.distance = max(Wm, Hm) * 1.2
        cam.azimuth, cam.elevation = 90, -89
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 1
    every = int(round((1 / 12) / model.opt.timestep))
    plan_every = int(0.5 / model.opt.timestep)

    # true corridor coverage: of the maze's FREE cells, how many are mapped free
    fg = info["free_grid"]
    free_world = [(x0 + (j + 0.5) * cell, y0 + (info["nrows"] - 1 - i + 0.5) * cell)
                  for i in range(fg.shape[0]) for j in range(fg.shape[1]) if fg[i, j]]
    free_ij = [grid.ij(wx, wy) for wx, wy in free_world]

    def maze_cov():
        return float(np.mean([grid.L[gi, gj] < -0.5 for gi, gj in free_ij]))

    # ---- WALL-FOLLOWING: each drone keeps a wall on one side (half LEFT-hand,
    # half RIGHT-hand) and follows it -> guaranteed to traverse the maze. ----
    def lidar_dir(d, phi):                            # range in world direction phi
        idx = int(round((phi % (2 * np.pi)) / (2 * np.pi) * nr)) % nr
        r = data.sensordata[d * nr + idx]
        return r if 0 < r <= max_range else max_range

    head = np.zeros(nd)
    hand = np.array([1.0 if d % 2 == 0 else -1.0 for d in range(nd)])   # L / R hand
    sense()
    for d in range(nd):                               # face the most open direction
        head[d] = max(angles, key=lambda a: lidar_dir(d, a))
    FRONT, TARGET, FAR = 0.65 * cell, 0.42 * cell, 0.95 * cell
    wall_hits = 0
    for k in range(int(args.seconds / model.opt.timestep)):
        if k % 8 == 0:
            sense(); discover()
        if k % int(10 / model.opt.timestep) == 0:
            mv = sum(np.linalg.norm(data.qvel[sdof[d]:sdof[d] + 2]) > 0.3 for d in range(nd))
            print(f"  t={data.time:4.0f}s  cov={100 * maze_cov():3.0f}%  "
                  f"moving={mv}/{nd}  found={len(found)}/{len(goal_xy)}")
        for d in range(nd):
            p = data.xpos[sb[d]]
            th, h = head[d], hand[d]
            front = min(lidar_dir(d, th), lidar_dir(d, th + 0.35), lidar_dir(d, th - 0.35))
            side = lidar_dir(d, th + h * np.pi / 2)
            if front < FRONT:
                dth = -h * 0.025                      # wall ahead -> turn away from wall side
            elif side > FAR:
                dth = h * 0.022                       # wall-side opened -> follow around corner
            else:
                dth = h * 0.010 * np.clip(side - TARGET, -1.5, 1.5)   # hug the wall
            head[d] = th + np.clip(dth, -0.03, 0.03)
            tgt = np.array([p[0] + 2.2 * np.cos(head[d]), p[1] + 2.2 * np.sin(head[d]), sz])
            f = 5.5 * (tgt - p) - 4.5 * data.qvel[sdof[d]:sdof[d] + 3]
            rngd = data.sensordata[d * nr:(d + 1) * nr]
            for i in range(nr):                       # reactive obstacle avoidance (safety)
                if 0 < rngd[i] < 1.4:
                    push = 18.0 * (1.4 - rngd[i]) / 1.4
                    f[0] -= push * np.cos(angles[i]); f[1] -= push * np.sin(angles[i])
            for e in range(nd):                       # separation
                if e != d:
                    dv = p[:2] - data.xpos[sb[e]][:2]; dn = np.hypot(*dv)
                    if 1e-3 < dn < 1.8:
                        f[:2] += 13.0 * dv / dn
            f[2] += 0.3 * 9.81
            mag = np.linalg.norm(f)
            if mag > 16.0:
                f *= 16.0 / mag
            data.xfrc_applied[sb[d], :3] = f
        mujoco.mj_step(model, data)
        for ci in range(data.ncon):
            c = data.contact[ci]
            ind = [d for d in range(nd) if c.geom1 in bg[d] or c.geom2 in bg[d]]
            if ind:
                other = c.geom2 if c.geom1 in bg[ind[0]] else c.geom1
                if other != ground and not any(other in bg[e] for e in range(nd)):
                    wall_hits += 1
        if args.gif and k % every == 0:
            renderer.update_scene(data, camera=cam, scene_option=opt)
            scene = Image.fromarray(renderer.render())
            mp = Image.fromarray(grid.image()).resize((560, 560), Image.NEAREST).convert("RGB")
            combo = Image.new("RGB", (1120, 560))
            combo.paste(scene, (0, 0)); combo.paste(mp, (560, 0))
            frames.append(combo)
        if len(found) == len(goal_xy) and maze_cov() > 0.97:
            break

    cov = maze_cov()                                 # true corridor coverage
    print(f"\nUNKNOWN PointMaze explored: {100 * cov:.0f}% mapped, "
          f"drop zones found {len(found)}/{len(goal_xy)}, "
          f"wall hits {wall_hits}, t={data.time:.0f}s")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    from PIL import Image
    Image.fromarray(grid.image()).resize((600, 600), Image.NEAREST).save(
        os.path.join(ROOT, "results", "pointmaze_map.png"))
    print("wrote results/pointmaze_map.png")
    if args.gif and frames:
        frames = frames[::2]
        pal = frames[len(frames) // 2].quantize(colors=128)
        q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
        q[0].save(os.path.join(ROOT, "results", "pointmaze.gif"), save_all=True,
                  append_images=q[1:], duration=110, loop=0, optimize=True)
        print(f"wrote results/pointmaze.gif ({len(q)} frames)")


if __name__ == "__main__":
    main()
