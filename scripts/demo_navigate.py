"""
Milestone 4 (navigation) demo — coordinate a swarm through a walled slalom.

The swarm must hold a compact formation and weave through two offset gates to
reach the goal:

    python scripts/demo_navigate.py --drones 6              # viewer (use mjpython on macOS)
    python scripts/demo_navigate.py --drones 6 --headless   # CI check

A moving formation centroid follows the gate waypoints at a fixed speed; each
drone tracks centroid + its slot in a small circle. The gates' gaps are only ~3 m
wide, so the formation has to stay tight to fit through.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_navigation_scene
from control.pd_controller import SwarmPD


def slot_offsets(n, radius=1.15):
    """Circle of formation slots (xy offsets), z=0. Radius chosen so neighbouring
    drones sit ~1.2 m apart (safe separation) for a 6-drone ring."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", type=int, default=6)
    ap.add_argument("--speed", type=float, default=0.7, help="centroid speed (m/s)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=34.0)
    args = ap.parse_args()

    xml, info = build_navigation_scene(n_drones=args.drones)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ctrl = SwarmPD(model, args.drones, kp=6.0, kd=4.5, fmax=10.0)

    nav_z = info["nav_z"]
    offsets = slot_offsets(args.drones)
    waypoints = [np.array([x, y, nav_z]) for (x, y) in info["waypoints"]]
    goal = np.array([info["goal"][0], info["goal"][1], nav_z])

    centroid = np.array([info["start"][0], info["start"][1], nav_z], dtype=float)
    wp_i = 0
    dt = model.opt.timestep

    print(f"drones={args.drones}  start={info['start']}  goal={info['goal']}")
    print(f"waypoints (weave through gates): {info['waypoints']}")

    def step():
        nonlocal wp_i, centroid
        # Cohesion gate: only advance the centroid while the formation is keeping
        # up. If a drone lags (e.g. snagged on a wall), the centroid waits so it
        # never drags the swarm through geometry.
        pos = ctrl.positions(data)
        # Cohesion = how far each drone is from its own slot (not the centroid),
        # so it's independent of formation size.
        slot_err = np.linalg.norm(pos[:, :2] - (centroid + offsets)[:, :2], axis=1)
        cohesive = slot_err.max() < 0.6
        target_wp = waypoints[wp_i]
        to = target_wp - centroid
        dist = np.linalg.norm(to)
        if dist < 0.25 and wp_i < len(waypoints) - 1:
            wp_i += 1
        elif cohesive and dist > 1e-6:             # advance centroid at fixed speed
            centroid = centroid + to / dist * min(args.speed * dt, dist)
        ctrl.apply(data, centroid + offsets)
        mujoco.mj_step(model, data)

    if args.headless:
        for _ in range(int(args.seconds / dt)):
            step()
        pos = ctrl.positions(data)
        d_goal = np.linalg.norm(pos[:, :2] - goal[:2], axis=1)
        ncon = data.ncon
        print(f"reached waypoint {wp_i + 1}/{len(waypoints)}")
        print(f"drone-to-goal dist: mean {d_goal.mean():.2f} m | max {d_goal.max():.2f} m")
        print(f"active contacts at end: {ncon}")
        print("PASS" if d_goal.max() < 1.5 else "NOT ARRIVED")
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step()
                viewer.sync()


if __name__ == "__main__":
    main()
