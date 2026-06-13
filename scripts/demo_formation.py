"""
Milestone 1 demo — drive the swarm into a named formation.

    python scripts/demo_formation.py --formation star            # opens viewer
    python scripts/demo_formation.py --formation star --headless  # no GUI, prints error

This is the LLM->targets->control loop with the LLM stubbed by formations.make().
Once your Gemini key is set, the commander will pick the formation name + params.
"""
import sys
import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_scene
from control.pd_controller import SwarmPD
from llm import formations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formation", default="star", choices=list(formations.REGISTRY))
    ap.add_argument("--drones", type=int, default=10)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    xml = build_scene(n_drones=args.drones, with_payload=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    ctrl = SwarmPD(model, args.drones)
    targets = formations.make(args.formation, args.drones)

    def step():
        ctrl.apply(data, targets)
        mujoco.mj_step(model, data)

    if args.headless:
        n_steps = int(args.seconds / model.opt.timestep)
        for _ in range(n_steps):
            step()
        err = np.linalg.norm(ctrl.positions(data) - targets, axis=1)
        print(f"formation={args.formation} drones={args.drones}")
        print(f"mean target error: {err.mean():.3f} m | max: {err.max():.3f} m")
        print("PASS" if err.max() < 0.25 else "NOT CONVERGED")
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step()
                viewer.sync()


if __name__ == "__main__":
    main()
