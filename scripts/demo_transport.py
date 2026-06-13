"""
Milestone 3 (Layer 2) demo — cooperative transport with suction-cup pickup.

The job is auto-sized: plan_transport() works out how heavy the tetromino is and
how many drones it needs. The carriers then run a full pickup sequence:

    approach (hover over the block)  ->  descend (lower the suction cups)
        ->  suction ON (grip)  ->  lift  ->  carry to the delivery point

    python scripts/demo_transport.py --shape Z                 # viewer (mjpython on macOS)
    python scripts/demo_transport.py --shape Z --headless      # CI check
    python scripts/demo_transport.py --shape L --to 3 2 --height 2.5 --headless
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_transport_scene, plan_transport
from control.pd_controller import SwarmPD

# Phase boundaries (seconds): approach, descend, lift, carry. Deliberately slow
# so the payload accelerates gently and barely tilts.
T_APPROACH, T_DESCEND, T_LIFT, T_CARRY = 2.0, 5.0, 8.5, 18.0
APPROACH_H = 0.8


def phase_name(t):
    return ("approach", "descend", "lift", "carry")[
        int(np.searchsorted([T_APPROACH, T_DESCEND, T_LIFT], t)) if t < T_CARRY else 3]


def payload_tilt_deg(data, body_id):
    """Tilt of the payload's local +z away from world +z, in degrees."""
    R = data.xmat[body_id].reshape(3, 3)
    cos = np.clip(R[2, 2], -1.0, 1.0)          # body-z . world-z
    return float(np.degrees(np.arccos(cos)))


def _rot(offsets, yaw):
    """Rotate (n,2) grip offsets by yaw (rad) about the payload center."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return offsets @ R.T


def make_transport_stepper(model, data, info, deliver, height,
                           path=None, yaw_of=None, yaw_rate=0.6, kp=10.0, kd=6.0,
                           hold_until=0.0, turn_slow=0.3):
    """Set up the carriers + suction grips and return a ``step()`` closure that
    runs approach -> descend -> grip -> lift -> carry. ``path(a)`` maps carry
    progress a in [0,1] to a payload-center xy; ``yaw_of(a)`` optionally maps it
    to a target payload yaw (rad) — rotating the grip formation turns the rigid
    slab, so the swarm can spin the payload 90 deg to fit a narrow gap. Shared by
    the demo, the renderer and the combined mission."""
    n = info["n_carriers"]
    half_z, tether = info["half_z"], info["tether_len"]
    origin = np.array(info["origin"])
    deliver = np.array(deliver)
    offsets = np.array([(ox, oy) for (ox, oy) in info["offsets"]])
    contact_z = info["contact_z"]
    grounded_cz = info["payload_z"]
    drone_z = lambda cz: cz + half_z + tether
    if path is None:
        path = lambda a: origin + a * (deliver - origin)
    if yaw_of is None:
        yaw_of = lambda a: 0.0

    ctrl = SwarmPD(model, n, kp=kp, kd=kd)
    grips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip{i}")
             for i in range(n)]
    for i in range(n):                            # start at approach height
        qadr = model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
        data.qpos[qadr + 2] += APPROACH_H
    mujoco.mj_forward(model, data)
    state = {"gripped": False, "yaw": 0.0, "prog": 0.0}
    base_rate = 1.0 / (T_CARRY - T_LIFT)          # carry progress per second

    def carry_a():
        return state["prog"]

    def targets_now():
        t = data.time
        rot_off = _rot(offsets, state["yaw"])
        if t < T_APPROACH:
            z = contact_z + APPROACH_H
            xy = origin + rot_off
        elif t < T_DESCEND:
            a = (t - T_APPROACH) / (T_DESCEND - T_APPROACH)
            z = contact_z + (1 - a) * APPROACH_H
            xy = origin + rot_off
        elif t < T_LIFT:
            a = (t - T_DESCEND) / (T_LIFT - T_DESCEND)
            z = drone_z(grounded_cz + a * (height - grounded_cz))
            xy = origin + rot_off
        else:
            xy = path(carry_a()) + rot_off
            z = drone_z(height)
        return np.column_stack([xy, np.full(n, z)])

    def step():
        if not state["gripped"] and data.time >= T_DESCEND:
            for eq in grips:
                data.eq_active[eq] = 1            # suction ON
            ctrl.ff_mass = info["share_per_drone"]
            state["gripped"] = True
        # Couriers only start carrying once t passes T_LIFT AND hold_until (e.g.
        # after the scout has finished mapping the whole maze).
        if state["gripped"] and data.time >= max(T_LIFT, hold_until):
            target = yaw_of(state["prog"])
            dyaw = np.clip(target - state["yaw"], -yaw_rate * model.opt.timestep,
                           yaw_rate * model.opt.timestep)
            state["yaw"] += dyaw
            # Nearly STOP forward motion while turning so the payload rotates in
            # place (centred in the open cell) instead of drifting into a wall.
            turning = abs(target - state["yaw"]) > 0.05
            rate = base_rate * (turn_slow if turning else 1.0)
            state["prog"] = min(1.0, state["prog"] + rate * model.opt.timestep)
        ctrl.apply(data, targets_now())
        mujoco.mj_step(model, data)

    return step, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    ap.add_argument("--to", nargs=2, type=float, default=[3.0, 0.0],
                    metavar=("X", "Y"), help="delivery point")
    ap.add_argument("--height", type=float, default=2.0, help="carry height (m)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=19.5)
    args = ap.parse_args()

    plan = plan_transport(shape=args.shape)
    print(f"PAYLOAD  '{args.shape}': {plan['n_cells']} cubes x {plan['mass_per_cube']:.2f} kg "
          f"= {plan['payload_mass']:.2f} kg  (cube {plan['cube_edge']} m, "
          f"rho {plan['density']} kg/m^3)")
    print(f"DRONES   need {plan['n_drones']} (>= {plan['n_cells']} for support, "
          f"capacity {plan['lift_per_drone']} kg/drone x{plan['margin']} margin) "
          f"-> {plan['share_per_drone']:.2f} kg each")

    xml, info = build_transport_scene(plan=plan)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")

    step, st = make_transport_stepper(model, data, info, args.to, args.height)
    print(f"SPACING  one drone per tile, vertical suction -> "
          f"{info['min_drone_sep']:.2f} m apart (center-to-center)")

    if args.headless:
        last, max_tilt = None, 0.0
        for _ in range(int(args.seconds / model.opt.timestep)):
            step()
            max_tilt = max(max_tilt, payload_tilt_deg(data, pid))
            phase = phase_name(data.time)
            if phase != last:
                tag = "  [suction ON]" if phase == "lift" else ""
                print(f"  t={data.time:4.1f}s  phase: {phase}{tag}")
                last = phase
        p = data.xpos[pid].copy()
        xy_err = float(np.linalg.norm(p[:2] - np.array(args.to)))
        z_err = abs(float(p[2]) - args.height)
        print(f"payload final: x={p[0]:.2f} y={p[1]:.2f} z={p[2]:.2f}")
        print(f"delivery xy error: {xy_err:.3f} m | height error: {z_err:.3f} m")
        print(f"max payload tilt: {max_tilt:.1f} deg (limit 15)")
        ok = xy_err < 0.4 and z_err < 0.4 and max_tilt < 15.0
        print("PASS" if ok else "NOT DELIVERED / TOO MUCH TILT")
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step()
                viewer.sync()


if __name__ == "__main__":
    main()
