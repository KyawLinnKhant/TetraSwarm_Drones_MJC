"""
Milestone 2 demo — natural language -> LLM commander -> swarm formation.

    # Gemini picks the formation + params from your words:
    python scripts/demo_commander.py -i "spread into a wide star"            # viewer
    python scripts/demo_commander.py -i "tight defensive ring" --headless    # CI check
    python scripts/demo_commander.py -i "flying wedge" --no-llm              # offline

This is the full Layer-1 loop: instruction -> Commander.plan() -> targets ->
SwarmPD -> MuJoCo. The commander validates the LLM's choice against the
formation registry, so the simulator only ever sees a legal command.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_scene
from control.pd_controller import SwarmPD
from llm.commander import Commander


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--instruction", default="form a tight defensive circle")
    ap.add_argument("--drones", type=int, default=10)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="force offline keyword fallback")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    commander = Commander(api_key="" if args.no_llm else Commander._AUTO)
    print(f"LLM commander: {'online' if commander.available else 'offline (fallback)'}")
    cmd = commander.plan(args.instruction, args.drones)
    print(f'  instruction : "{args.instruction}"')
    print(f"  -> formation: {cmd.formation}  (via {cmd.source})")
    print(f"  -> params   : {cmd.params or '(defaults)'}")
    print(f"  -> reasoning: {cmd.reasoning}")

    targets = cmd.targets()

    xml = build_scene(n_drones=args.drones, with_payload=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ctrl = SwarmPD(model, args.drones)

    def step():
        ctrl.apply(data, targets)
        mujoco.mj_step(model, data)

    if args.headless:
        for _ in range(int(args.seconds / model.opt.timestep)):
            step()
        err = np.linalg.norm(ctrl.positions(data) - targets, axis=1)
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
