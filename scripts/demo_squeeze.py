"""
Smart 'turn-to-fit' demo — rotate the payload 90 deg to clear a narrow doorway.

The Cleveland Z is wide (~4.5 m) and short (~3 m). The doorway gap is only ~3.5 m,
narrower than the Z is wide. The planner notices the slab won't fit, so the swarm
rotates it 90 deg (now ~3 m wide, ~4.5 m long) to slip its narrow side through,
then rotates back. Turning the Z turns the drones with it.

    python scripts/demo_squeeze.py               # viewer (mjpython on macOS)
    python scripts/demo_squeeze.py --headless    # CI check
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_squeeze_scene, plan_transport, payload_extent
from scripts.demo_transport import (make_transport_stepper, phase_name,
                                    payload_tilt_deg, T_LIFT, T_CARRY)

HALF_PI = np.pi / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    ap.add_argument("--gap", type=float, default=3.5, help="doorway gap width (m)")
    ap.add_argument("--height", type=float, default=2.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=27.0)
    args = ap.parse_args()

    plan = plan_transport(shape=args.shape)
    xml, info = build_squeeze_scene(plan=plan, gap_w=args.gap)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")

    start = np.array(info["start"])
    goal = np.array(info["goal"])
    ex, ey = payload_extent(info)                 # half-extents (x wide, y short)
    wall_y, gap_w = info["wall_y"], info["gap_w"]
    must_turn = 2 * ex > gap_w                    # too wide to fit as-is?

    print(f"PAYLOAD '{args.shape}': {2*ex:.1f} m wide x {2*ey:.1f} m long, "
          f"{plan['payload_mass']:.1f} kg, {info['n_carriers']} drones")
    print(f"DOORWAY gap {gap_w:.1f} m  ->  "
          + ("TOO NARROW: must turn 90 deg to fit" if must_turn
             else "fits without turning"))

    def path(a):
        return start + a * (goal - start)

    # The turn must FINISH before the slab's leading edge reaches the door,
    # because mid-rotation its diagonal is wider than the gap. So begin turning a
    # full slab-length early and hold the turn across the whole doorway band.
    band = ex + 3.0

    def yaw_of(a):
        y = path(a)[1]
        return HALF_PI if (must_turn and abs(y - wall_y) < band) else 0.0

    step, st = make_transport_stepper(model, data, info, goal, args.height,
                                      path=path, yaw_of=yaw_of, yaw_rate=1.6)

    if args.headless:
        last, max_tilt, turned = None, 0.0, False
        for _ in range(int(args.seconds / model.opt.timestep)):
            step()
            max_tilt = max(max_tilt, payload_tilt_deg(data, pid))
            turned = turned or abs(st["yaw"]) > HALF_PI * 0.8
            ph = phase_name(data.time)
            if ph != last:
                print(f"  t={data.time:4.1f}s  phase: {ph}")
                last = ph
        p = data.xpos[pid].copy()
        yaw_deg = np.degrees(st["yaw"])
        xy_err = float(np.linalg.norm(p[:2] - goal))
        print(f"payload final: x={p[0]:.2f} y={p[1]:.2f} z={p[2]:.2f}  "
              f"yaw now {yaw_deg:.0f} deg")
        print(f"did 90-deg turn at the door: {turned} | max tilt {max_tilt:.1f} deg")
        print(f"goal xy error: {xy_err:.3f} m")
        ok = xy_err < 0.5 and (turned or not must_turn) and max_tilt < 20
        print("PASS" if ok else "FAIL")
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step()
                viewer.sync()


if __name__ == "__main__":
    main()
