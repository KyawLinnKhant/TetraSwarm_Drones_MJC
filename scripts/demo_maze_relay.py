"""
Maze relay — one swarm shuttles four tetromino blocks from the maze's centre
depot out to its four corners, ROUTING THROUGH THE MAZE (no straight or diagonal
shortcut between corners). Each block is carried kinematically and turned to keep
its narrow side facing the way it's going.

    Z -> top-left,  I -> bottom-right,  O -> top-right,  L -> bottom-left

    python scripts/demo_maze_relay.py --headless
    python scripts/demo_maze_relay.py --render        # writes results/maze_relay.gif
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_maze_relay_scene
from control.pd_controller import SwarmPD
from scripts.demo_maze import arc_tools

LIFT_Z = 1.6


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="prepend the ViPER maze-clearing sweep before the relay")
    ap.add_argument("--sweep-gif", dest="sweep_gif", action="store_true",
                    help="render ONLY the sweep -> results/viper_maze.gif, then stop")
    args = ap.parse_args()

    xml, info = build_maze_relay_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ctrl = SwarmPD(model, 4, kp=90, kd=19)         # very stiff tracking -> low lag
    hz, tether = info["half_z"], info["tether_len"]
    ground_z = hz
    drone_above = lambda cz: cz + hz + tether

    qadr = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_free")]
            for s in info["blocks"]}
    vadr = {s: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_free")]
            for s in info["blocks"]}
    bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s) for s in info["blocks"]}
    pose = {s: (info["depot"][s][0], info["depot"][s][1], ground_z, 0.0) for s in info["blocks"]}
    # drone free-joint qpos addresses (drones are driven kinematically too)
    dqadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
             for i in range(4)]

    # Compact transit formation (2x2 square) the drones fly EMPTY in.
    SQ = np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    # Order each block's grips so drone-i's square slot maps to its NEAREST grip
    # (Hungarian assignment) -> no two drones swap/cross when morphing in.
    from scipy.optimize import linear_sum_assignment
    grips = {}
    for s in info["blocks"]:
        g = np.array(info["grips"][s])
        cost = np.linalg.norm(SQ[:, None, :] - g[None, :, :], axis=-1)
        _, col = linear_sum_assignment(cost)
        grips[s] = g[col]
    th = info["tile_half"]
    # long-axis offset: align the block's LONGER side with the heading so its
    # narrow side faces each doorway (local-y is the long axis for L).
    long_off, is_sym = {}, {}
    for s in info["blocks"]:
        g = grips[s]
        xext, yext = np.ptp(g[:, 0]) + 2 * th, np.ptp(g[:, 1]) + 2 * th
        long_off[s] = (np.pi / 2) if yext > xext else 0.0   # align LONG side w/ travel
        is_sym[s] = abs(xext - yext) < 0.3                  # square -> never rotate

    def write_blocks():
        for s in info["blocks"]:
            x, y, z, yaw = pose[s]
            data.qpos[qadr[s]:qadr[s] + 3] = [x, y, z]
            data.qpos[qadr[s] + 3:qadr[s] + 7] = yaw_quat(yaw)
            data.qvel[vadr[s]:vadr[s] + 6] = 0

    write_blocks()
    mujoco.mj_forward(model, data)

    # per-block wall-contact detection (blocks are SOLID now)
    bgeoms = {s: {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid[s]}
              for s in info["blocks"]}
    ground_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    wall_hits = {s: 0 for s in info["blocks"]}

    def count_hits():
        for ci in range(data.ncon):
            c = data.contact[ci]
            for s in info["blocks"]:
                in1, in2 = c.geom1 in bgeoms[s], c.geom2 in bgeoms[s]
                if in1 ^ in2:
                    other = c.geom2 if in1 else c.geom1
                    if other != ground_g:
                        wall_hits[s] += 1

    frames, renderer, cam, every, kc = [], None, None, 1, [0]
    if args.render:
        renderer = mujoco.Renderer(model, height=640, width=680)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0]
        span = max(info["nx"], info["ny"]) * info["cell"]
        cam.distance, cam.azimuth, cam.elevation = span * 1.45, 90, -89
        every = int(round((1 / 8) / model.opt.timestep))     # 8 fps, smaller file

    def gworld(goff, cx, cy, yaw, dz):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        return np.column_stack([np.array([cx, cy]) + goff @ R.T, np.full(len(goff), dz)])

    def cur_off(center):
        cur = np.array([data.qpos[dqadr[i]:dqadr[i] + 2] for i in range(4)])
        return cur - np.array(center)

    def order_grips(base, center, yaw):
        """Reorder base grip offsets so drone-i maps to its NEAREST (rotated) grip
        — no two drones swap/cross when they grip the block."""
        c, s = np.cos(yaw), np.sin(yaw)
        gw = base @ np.array([[c, -s], [s, c]]).T
        cost = np.linalg.norm(cur_off(center)[:, None] - gw[None], axis=-1)
        _, col = linear_sum_assignment(cost)
        return base[col]

    def step(targets, carry=None, cmd_center=None, cmd_yaw=0.0, cz=LIFT_Z):
        # Fully kinematic: drones snap to their commanded grip targets and the
        # block to the commanded pose, so drones + block are ALWAYS in sync (no
        # lag, no flicker) and follow the clean ideal route (no wall clipping).
        for i in range(4):
            a = dqadr[i]
            data.qpos[a:a + 3] = targets[i]
            data.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        if carry is not None:
            pose[carry] = (float(cmd_center[0]), float(cmd_center[1]), cz, cmd_yaw)
        write_blocks()
        mujoco.mj_forward(model, data)
        count_hits()
        if renderer is not None:
            if kc[0] % every == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())
            kc[0] += 1

    def hold_phase(seconds, center, yaw, z_from, z_to, goff, carry=None):
        n = int(seconds / model.opt.timestep)
        for k in range(n):
            a = (k + 1) / n
            cz = z_from + a * (z_to - z_from)
            step(gworld(goff, center[0], center[1], yaw, drone_above(cz)),
                 carry, cmd_center=center, cmd_yaw=yaw, cz=cz)

    def transit(route, speed=3.5):
        """Fly empty along a polyline, KEEPING the drones' current arrangement
        (rigid translation — no reassignment, so no swap/teleport)."""
        if len(route) < 2:
            return route[-1]
        path, _ = arc_tools(route)
        total = np.sum(np.linalg.norm(np.diff(np.array(route), axis=0), axis=1))
        base = speed / max(total, 1e-6)
        off0 = cur_off(path(0.0))                  # hold whatever square we're in
        prog, guard = 0.0, int(60 / model.opt.timestep)
        while prog < 1.0 and guard > 0:
            guard -= 1
            prog = min(1.0, prog + base * model.opt.timestep)
            cx, cy = path(prog)
            step(np.column_stack([np.array([cx, cy]) + off0, np.full(4, drone_above(LIFT_Z))]))
        return (cx, cy)

    def morph_to(center, target_off, seconds=1.2):
        """Glide drones from their current positions to ``target_off`` slots, each
        drone assigned to its NEAREST slot (no crossing at any angle)."""
        cost = np.linalg.norm(cur_off(center)[:, None] - target_off[None], axis=-1)
        _, col = linear_sum_assignment(cost)
        tgt = target_off[col]
        start = cur_off(center)
        n = int(seconds / model.opt.timestep)
        for k in range(n):
            a = (k + 1) / n
            off = (1 - a) * start + a * tgt
            step(np.column_stack([np.array(center) + off, np.full(4, drone_above(LIFT_Z))]))

    def follow(route, goff, carry, yaw0, speed=2.0, yaw_rate=0.6, yaw_offset=0.0,
               rotate=True):
        """Drive the formation along a polyline route, yaw=heading(+offset) when
        ``rotate`` (turning to fit); carry the block kinematically using fixed grip
        offsets ``goff``. Returns final (center, yaw)."""
        path, heading = arc_tools(route)
        prog, yaw = 0.0, yaw0
        total = np.sum(np.linalg.norm(np.diff(np.array(route), axis=0), axis=1))
        base = speed / max(total, 1e-6)
        guard = int(90 / model.opt.timestep)
        while prog < 1.0 and guard > 0:
            guard -= 1
            tgt = (heading(min(prog + 1e-3, 1.0)) + yaw_offset) if rotate else yaw0
            dyaw = np.clip(tgt - yaw, -yaw_rate * model.opt.timestep, yaw_rate * model.opt.timestep)
            yaw += dyaw
            turning = abs(tgt - yaw) > 0.05
            prog = min(1.0, prog + base * (0.03 if turning else 1.0) * model.opt.timestep)
            cx, cy = path(prog)
            step(gworld(goff, cx, cy, yaw, drone_above(LIFT_Z)),
                 carry, cmd_center=(cx, cy), cmd_yaw=yaw, cz=LIFT_Z)
        return (cx, cy), yaw

    ctr = tuple(info["depot_center"])

    # ---- optional ViPER SWEEP prologue: the 4 drones first fly the path that
    # marmotlab/ViPER planned to clear/map this same maze, then return to depot.
    # ViPER plans at node level (4 m), so we route each move through the CORRIDORS
    # (A* on a wall grid) instead of straight lines, and drive all drones at one
    # synced constant speed -> smooth purposeful sweep, no teleporting/jitter. ----
    if args.sweep:
        import heapq
        from collections import deque
        # the drop zones are UNKNOWN: hide the goal markers until a drone passes
        # near one during the sweep (then it lights up = discovered).
        goal_sid = {cn: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"goal_{cn}")
                    for cn in info["routes"]}
        goal_xy = {cn: np.array(model.site_pos[s][:2]) for cn, s in goal_sid.items()}
        for s in goal_sid.values():
            model.site_rgba[s] = [0.35, 0.35, 0.4, 0.06]   # undiscovered (near-invisible)
        found = set()

        def scan_drop_zones():
            dpos = np.array([data.qpos[dqadr[i]:dqadr[i] + 2] for i in range(4)])
            for cn, s in goal_sid.items():
                if cn not in found and np.linalg.norm(dpos - goal_xy[cn], axis=1).min() < 5.5:
                    found.add(cn)
                    model.site_rgba[s] = [0.15, 0.95, 0.35, 0.95]   # discovered!
                    print(f"  [sweep] discovered drop zone -> {cn}")

        tp = os.path.join(ROOT, "external", "ViPER", "results", "viper_traj.npz")
        vd = np.load(tp)
        vc = vd["traj"].astype(float)                 # (T,4,2) ViPER grid cells
        gt_w = float(vd["gt_shape"][1])
        nxc, cellc = info["nx"], info["cell"]
        x0 = -nxc * cellc / 2.0
        sxc = (nxc * cellc) / gt_w
        sweep_z = 1.0

        # occupancy grid from the maze walls (+1-cell clearance)
        res = 0.5
        gN = int(nxc * cellc / res) + 2
        occ = np.zeros((gN, gN), bool)
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        for g in range(model.ngeom):
            if g == gid or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX \
                    or model.geom_bodyid[g] != 0:
                continue
            cxw, cyw = data.geom_xpos[g][:2]
            hx, hy = model.geom_size[g][:2]
            r0, r1 = int((cyw - hy - x0) / res), int((cyw + hy - x0) / res)
            c0, c1 = int((cxw - hx - x0) / res), int((cxw + hx - x0) / res)
            occ[max(0, r0):r1 + 1, max(0, c0):c1 + 1] = True
        occd = occ.copy()
        occd[1:-1, 1:-1] |= (occ[2:, 1:-1] | occ[:-2, 1:-1] | occ[1:-1, 2:] | occ[1:-1, :-2])
        w2c = lambda wx, wy: (int((wy - x0) / res), int((wx - x0) / res))
        c2w = lambda r, c: (x0 + (c + 0.5) * res, x0 + (r + 0.5) * res)

        def snap(rc):
            if 0 <= rc[0] < gN and 0 <= rc[1] < gN and not occd[rc]:
                return rc
            q, seen = deque([rc]), {rc}
            while q:
                c = q.popleft()
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    n = (c[0] + dx, c[1] + dy)
                    if 0 <= n[0] < gN and 0 <= n[1] < gN and n not in seen:
                        if not occd[n]:
                            return n
                        seen.add(n); q.append(n)
            return rc

        def astar(s, gl):
            mv = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            h = lambda a: abs(a[0] - gl[0]) + abs(a[1] - gl[1])
            oh = [(h(s), 0.0, s)]
            came, gsc = {s: None}, {s: 0.0}
            while oh:
                _, gc, c = heapq.heappop(oh)
                if c == gl:
                    p = [c]
                    while came[p[-1]] is not None:
                        p.append(came[p[-1]])
                    return p[::-1]
                for dx, dy in mv:
                    n = (c[0] + dx, c[1] + dy)
                    if 0 <= n[0] < gN and 0 <= n[1] < gN and not occd[n]:
                        ng = gc + (1.4 if dx and dy else 1.0)
                        if ng < gsc.get(n, 1e9):
                            gsc[n] = ng; came[n] = c
                            heapq.heappush(oh, (ng + h(n), ng, n))
            return [s]

        # corridor-routed world path per drone (concatenate A* between waypoints)
        apath_pts = []
        for i in range(4):
            wpts = [snap(w2c(x0 + vc[t, i, 0] * sxc, x0 + vc[t, i, 1] * sxc))
                    for t in range(len(vc))]
            pts = [c2w(*wpts[0])]
            for t in range(len(wpts) - 1):
                for rc in astar(wpts[t], wpts[t + 1])[1:]:
                    pts.append(c2w(*rc))
            apath_pts.append(pts)

        # COVERAGE COMPLETION: ViPER's pretrained policy leaves ~26% of this
        # (out-of-distribution) maze unswept; route the drones through the cells it
        # missed so the whole arena gets mapped before transport.
        samp = 8
        covered = np.array([p for pts in apath_pts for p in pts])
        free_centers = [(r, c) for r in range(2, gN - 2, samp) for c in range(2, gN - 2, samp)
                        if not occd[r, c]]
        todo = [(r, c) for (r, c) in free_centers
                if np.linalg.norm(covered - np.array(c2w(r, c)), axis=1).min() > 2.6]
        assign = {i: [] for i in range(4)}
        for rc in todo:                                  # give each missed cell to nearest drone
            w = np.array(c2w(*rc))
            i = int(np.argmin([np.linalg.norm(np.array(apath_pts[k][-1]) - w) for k in range(4)]))
            assign[i].append(rc)
        for i in range(4):
            cur = w2c(*apath_pts[i][-1])
            rem = assign[i]
            while rem:
                j = int(np.argmin([abs(cur[0] - r[0]) + abs(cur[1] - r[1]) for r in rem]))
                goal = rem.pop(j)
                for rc in astar(cur, goal)[1:]:
                    apath_pts[i].append(c2w(*rc))
                cur = goal

        apaths = []
        for pts in apath_pts:
            p = np.array(pts)
            cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
            apaths.append((p, cum, max(cum[-1], 1e-6)))

        def at(i, frac):                              # point at arc-fraction along path i
            p, cum, total = apaths[i]
            dd = frac * total
            j = int(np.clip(np.searchsorted(cum, dd) - 1, 0, len(p) - 2))
            seg = cum[j + 1] - cum[j]
            a = (dd - cum[j]) / seg if seg > 1e-9 else 0.0
            return p[j] + a * (p[j + 1] - p[j])

        ncall = int(45.0 / model.opt.timestep)        # ~45 s sweep (ViPER paths + completion)
        for k in range(ncall):
            frac = (k + 1) / ncall
            step(np.array([[*at(i, frac), sweep_z] for i in range(4)]))
            if k % 20 == 0:
                scan_drop_zones()                     # discover drop zones en route
        print(f"  [sweep] drop zones found: {len(found)}/{len(goal_sid)} -> {sorted(found)}")
        # fly back to the depot-centre square so the relay can begin cleanly
        cur0 = np.array([data.qpos[dqadr[i]:dqadr[i] + 2] for i in range(4)])
        nret = int(2.5 / model.opt.timestep)
        for k in range(nret):
            a = (k + 1) / nret
            tg = [[*(cur0[i] + a * ((np.array(ctr) + SQ[i]) - cur0[i])),
                   drone_above(LIFT_Z)] for i in range(4)]
            step(np.array(tg))

        if args.sweep_gif and args.render and frames:   # save just the sweep, then stop
            from PIL import Image
            renderer.close()
            imgs = [Image.fromarray(f) for f in frames]
            pal = imgs[len(imgs) // 2].quantize(colors=128)
            q = [im.quantize(palette=pal, dither=Image.NONE) for im in imgs]
            out = os.path.join(ROOT, "results", "viper_maze.gif")
            q[0].save(out, save_all=True, append_images=q[1:], duration=90, loop=0, optimize=True)
            print(f"wrote {out}  ({len(q)} sweep frames)")
            return

    # start the drones in the compact square at the depot centre
    for i in range(4):
        data.qpos[dqadr[i]:dqadr[i] + 2] = np.array(ctr) + SQ[i]
        data.qpos[dqadr[i] + 2] = drone_above(LIFT_Z)
    mujoco.mj_forward(model, data)

    log = []
    cur = ctr
    for s in info["blocks"]:
        corner = info["deliver"][s]
        dep = tuple(info["depot"][s])
        route = info["routes"][corner]               # depot-cell -> corner
        # fly EMPTY to the block (keep arrangement), then assign drones to the
        # NEAREST grips and glide in (no swap/teleport)
        transit([cur, dep], speed=3.0)
        goff = order_grips(grips[s], dep, 0.0)
        morph_to(dep, goff)
        # descend, grip + lift
        hold_phase(1.0, dep, 0.0, LIFT_Z, ground_z, goff)
        hold_phase(1.2, dep, 0.0, ground_z, LIFT_Z, goff, carry=s)
        # translate to the open centre cell at yaw 0 (long bar rotates in clear)
        nT = int(1.2 / model.opt.timestep)
        for k in range(nT):
            a = (k + 1) / nT
            tx, ty = dep[0] + a * (ctr[0] - dep[0]), dep[1] + a * (ctr[1] - dep[1])
            step(gworld(goff, tx, ty, 0.0, drone_above(LIFT_Z)),
                 s, cmd_center=(tx, ty), cmd_yaw=0.0, cz=LIFT_Z)
        # carry through the maze, rotating to fit (square block never rotates)
        c, yaw = follow(list(route), goff, s, 0.0, yaw_offset=long_off[s],
                        rotate=not is_sym[s])
        hold_phase(1.2, c, yaw, LIFT_Z, ground_z, goff, carry=s)
        bx, by = data.xpos[bid[s]][:2]            # leave the block where it landed
        pose[s] = (float(bx), float(by), ground_z, yaw)
        # release: drones glide to the NEAREST square slot (no crossing), then
        # fly EMPTY back through the maze keeping that arrangement
        morph_to(c, SQ)
        transit(list(route)[::-1] + [ctr], speed=4.0)
        cur = ctr
        d = float(np.linalg.norm(np.array(data.xpos[bid[s]][:2]) - np.array(info["routes"][corner][-1])))
        log.append(d)
        print(f"delivered {s} -> {corner}  (err {d:.2f} m, wall-hits {wall_hits[s]})")

    clean = all(h == 0 for h in wall_hits.values())
    print(f"wall hits per block: {wall_hits}")
    print("PASS" if all(d < 1.0 for d in log) and clean else "FAIL (block hit a wall)")
    if args.render and frames:
        from PIL import Image
        renderer.close()
        gif = os.path.join(ROOT, "results", "maze_relay.gif")
        # Build a palette that EXPLICITLY contains the Tetris block colours (as a
        # swatch band appended to a seed frame) so the small blocks never get
        # quantised to grey; dither so they survive on the dark floor.
        seed = frames[min(4, len(frames) - 1)]
        W = seed.shape[1]
        swatch = np.zeros((20, W, 3), np.uint8)
        for i, c in enumerate([(237, 28, 36), (0, 204, 242),
                               (250, 224, 13), (242, 102, 0)]):   # Z, I, O, L
            swatch[:, i * (W // 4):(i + 1) * (W // 4)] = c
        pal = Image.fromarray(np.vstack([seed, swatch])).quantize(colors=220)
        imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.FLOYDSTEINBERG)
                for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)
        Image.fromarray(frames[len(frames) // 2]).save(os.path.join(ROOT, "results", "maze_relay.png"))
        print(f"wrote {gif}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
