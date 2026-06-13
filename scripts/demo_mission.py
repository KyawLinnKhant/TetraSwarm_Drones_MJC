"""
Mission demo (Layer 2 + Layer 5 combined) — pick up and DELIVER through a maze.

The swarm sizes the job, grips a payload at the start zone with suction cups,
then navigates it through the walled slalom to the goal:

    grip  ->  lift  ->  weave through gate A and gate B  ->  set down at goal

    python scripts/demo_mission.py --shape Z              # viewer (mjpython on macOS)
    python scripts/demo_mission.py --shape Z --headless   # CI check
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_mission_scene, plan_transport
from scripts.demo_transport import make_transport_stepper, phase_name, T_CARRY


def polyline(points):
    """Return f(a) mapping a in [0,1] to a point along the polyline by arc length."""
    pts = np.array(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]

    def f(a):
        d = np.clip(a, 0, 1) * total
        i = int(np.searchsorted(cum, d) - 1)
        i = max(0, min(i, len(seg) - 1))
        t = (d - cum[i]) / seg[i] if seg[i] > 1e-9 else 0.0
        return pts[i] + t * (pts[i + 1] - pts[i])

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    ap.add_argument("--height", type=float, default=2.0, help="carry height (m)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=19.5)
    args = ap.parse_args()

    plan = plan_transport(shape=args.shape)
    print(f"MISSION  carry '{args.shape}' ({plan['payload_mass']:.2f} kg) with "
          f"{plan['n_drones']} drones through the slalom to the goal")

    xml, info = build_mission_scene(plan=plan)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")

    start, goal = info["start"], info["goal"]
    route = polyline([start] + list(info["waypoints"]))   # start -> gates -> goal
    step, _ = make_transport_stepper(model, data, info, goal, args.height,
                                     path=route)

    # "Viper" scout: flies the route first (t in [0, T_SCOUT]) to map it, then
    # holds station at the goal while the couriers carry the payload through.
    T_SCOUT = 7.0
    scout_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scout")
    scout_dof = model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "scout_free")]
    scout_z = info["scout_z"]

    def drive_scout():
        a = min(1.0, data.time / T_SCOUT)
        tx, ty = route(a)
        tgt = np.array([tx, ty, scout_z])
        pos = data.xpos[scout_bid]
        vel = data.qvel[scout_dof:scout_dof + 3]
        f = 8.0 * (tgt - pos) - 5.0 * vel
        f[2] += 0.3 * 9.81
        data.xfrc_applied[scout_bid, :3] = f

    if args.headless:
        last = None
        for _ in range(int(args.seconds / model.opt.timestep)):
            drive_scout()
            step()
            ph = phase_name(data.time)
            scouting = "  [scout mapping]" if data.time < T_SCOUT else ""
            if ph != last:
                print(f"  t={data.time:4.1f}s  phase: {ph}{scouting}"
                      + ("  [suction ON]" if ph == "lift" else "")
                      + ("  weaving through gates" if ph == "carry" else ""))
                last = ph
        p = data.xpos[pid].copy()
        xy_err = float(np.linalg.norm(p[:2] - np.array(goal)))
        print(f"payload final: x={p[0]:.2f} y={p[1]:.2f} z={p[2]:.2f}")
        print(f"goal xy error: {xy_err:.3f} m")
        print("PASS" if xy_err < 0.5 else "NOT DELIVERED")
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                drive_scout()
                step()
                viewer.sync()


if __name__ == "__main__":
    main()
