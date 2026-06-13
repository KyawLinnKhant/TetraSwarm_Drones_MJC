"""
Maze courier demo — carry the Cleveland Z through a braided (no-dead-end) maze.

The couriers follow the route and keep the Z's NARROW side facing the way it's
going (yaw = travel heading), so when the route turns a corner the swarm turns
the slab 90 deg to thread the next doorway. (Checkpoint 1 of the VIPER mission;
the scout maps the route in Checkpoint 2.)

    python scripts/demo_maze.py               # viewer (mjpython on macOS)
    python scripts/demo_maze.py --headless    # CI check
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_maze_scene, plan_transport, payload_extent
from scripts.demo_transport import make_transport_stepper, phase_name, payload_tilt_deg


def arc_tools(points):
    """Return (path(a), heading(a)) over the polyline by arc length."""
    pts = np.array(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    dirs = np.diff(pts, axis=0) / np.maximum(seg, 1e-9)[:, None]

    def idx(a):
        d = np.clip(a, 0, 1) * total
        i = int(np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1))
        return i, (d - cum[i]) / seg[i] if seg[i] > 1e-9 else 0.0

    def path(a):
        i, t = idx(a)
        return pts[i] + t * (pts[i + 1] - pts[i])

    def heading(a):
        i, _ = idx(a)
        return np.arctan2(dirs[i, 1], dirs[i, 0])

    return path, heading


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=66.0)
    args = ap.parse_args()

    plan = plan_transport(shape=args.shape)
    xml, info = build_maze_scene(seed=args.seed, plan=plan)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")

    path, heading = arc_tools(info["route"])
    ex, ey = payload_extent(info)
    print(f"MAZE route {len(info['route'])} cells; Z is {2*ex:.1f}x{2*ey:.1f} m, "
          f"{plan['payload_mass']:.1f} kg, {info['n_carriers']} drones")
    print("couriers keep the narrow side forward (yaw = heading), turning at corners")

    # Scout (Viper) maps the WHOLE maze first (boustrophedon coverage), and only
    # then do the couriers start carrying (hold_until = T_MAP).
    T_MAP = 20.0
    sweep = np.array(info["sweep"])
    # Align the slab's actual LONG axis with travel (short side faces each
    # doorway). Long axis is local-x for most shapes but local-y for L, so add a
    # 90° offset when the y-extent is larger.
    long_off = (np.pi / 2) if ey > ex else 0.0
    yaw_of = lambda a: heading(a) + long_off
    step, st = make_transport_stepper(model, data, info, info["goal"], args.height,
                                      path=path, yaw_of=yaw_of, yaw_rate=0.8,
                                      hold_until=T_MAP, turn_slow=0.05)

    sb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scout")
    sdof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scout_free")]
    sz = info["scout_z"]

    def drive_scout():
        a = min(0.999, data.time / T_MAP)         # sweep all cells over [0, T_MAP]
        seg = a * (len(sweep) - 1)
        i = int(seg)
        tx, ty = sweep[i] + (seg - i) * (sweep[i + 1] - sweep[i])
        f = 11.0 * (np.array([tx, ty, sz]) - data.xpos[sb]) - 5.0 * data.qvel[sdof:sdof + 3]
        f[2] += 0.3 * 9.81
        data.xfrc_applied[sb, :3] = f

    # payload geoms + ground geom, to detect payload<->WALL contact (not ground)
    pgeoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == pid}
    ground_g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    if args.headless:
        last, max_tilt, wall_hits = None, 0.0, 0
        for _ in range(int(args.seconds / model.opt.timestep)):
            drive_scout()
            step()
            max_tilt = max(max_tilt, payload_tilt_deg(data, pid))
            for ci in range(data.ncon):
                c = data.contact[ci]
                other = (c.geom2 if c.geom1 in pgeoms else c.geom1)
                if ((c.geom1 in pgeoms) ^ (c.geom2 in pgeoms)) and other != ground_g:
                    wall_hits += 1                # payload vs a WALL (not the floor)
            ph = phase_name(data.time)
            if ph != last:
                print(f"  t={data.time:4.1f}s  phase: {ph}")
                last = ph
        p = data.xpos[pid].copy()
        xy_err = float(np.linalg.norm(p[:2] - np.array(info["goal"])))
        print(f"payload final: x={p[0]:.2f} y={p[1]:.2f}  goal err {xy_err:.2f} m | "
              f"max tilt {max_tilt:.1f} deg | payload-wall contacts {wall_hits}")
        print("PASS" if xy_err < 0.6 and max_tilt < 20 and wall_hits == 0 else "FAIL")
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                drive_scout()
                step()
                viewer.sync()


if __name__ == "__main__":
    main()
