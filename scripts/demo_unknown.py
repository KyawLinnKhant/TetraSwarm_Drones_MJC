"""
Unknown-payload cooperative transport (à la Swarm_Drones_ROSCS).

The carriers are told NOTHING about the payload: a random tetromino with a random
mass is dropped in, and the swarm lifts it while estimating the total mass online
with the adaptive feedforward controller. Writes the transport graphs.

    python scripts/demo_unknown.py                 # random shape+mass, prints estimate
    python scripts/demo_unknown.py --graphs        # also writes results/figures/transport{1,2}.png
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_transport_scene, plan_transport
from control.adaptive import AdaptiveLift

T_APPROACH, T_DESCEND, T_LIFT = 1.5, 3.5, 7.0
APPROACH_H = 0.8


def run(shape, density, height=2.0, seconds=16.0, log=False):
    """Lift an unknown tetromino; return (true_mass, est_history, info)."""
    # The SCENE knows the true mass; the CONTROLLER is given none of it.
    plan = plan_transport(shape=shape, density=density)
    xml, info = build_transport_scene(plan=plan)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    n = info["n_carriers"]
    pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    half_z, tether = info["half_z"], info["tether_len"]
    origin = np.array(info["origin"])
    offsets = np.array(info["offsets"])
    contact_z, grounded = info["contact_z"], info["payload_z"]
    drone_z = lambda cz: cz + half_z + tether

    ctrl = AdaptiveLift(model, n, kp=10, kd=6, gamma=14, fmax=None)
    grips = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip{i}")
             for i in range(n)]
    for i in range(n):
        qa = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
        data.qpos[qa + 2] += APPROACH_H
    mujoco.mj_forward(model, data)

    ts, est, true_m, pz, gripped = [], [], plan["payload_mass"], [], False
    for _ in range(int(seconds / model.opt.timestep)):
        t = data.time
        if t < T_APPROACH:
            z = contact_z + APPROACH_H
        elif t < T_DESCEND:
            z = contact_z + (1 - (t - T_APPROACH) / (T_DESCEND - T_APPROACH)) * APPROACH_H
        elif t < T_LIFT:
            a = (t - T_DESCEND) / (T_LIFT - T_DESCEND)
            z = drone_z(grounded + a * (height - grounded))
        else:
            z = drone_z(height)
        if not gripped and t >= T_DESCEND:
            for eq in grips:
                data.eq_active[eq] = 1
            ctrl.active = True
            gripped = True
        targets = np.column_stack([origin + offsets, np.full(n, z)])
        ctrl.apply(data, targets)
        mujoco.mj_step(model, data)
        ts.append(t)
        est.append(ctrl.estimated_payload_mass())
        pz.append(float(data.xpos[pid][2]))
    return true_m, np.array(ts), np.array(est), np.array(pz), ctrl, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default=None, help="default: random")
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    shape = args.shape or rng.choice(list("IOTLSZ"))
    density = float(rng.uniform(8, 22))           # unknown to the controller

    true_m, ts, est, pz, ctrl, info = run(shape, density)
    print(f"UNKNOWN payload: shape='{shape}'  true mass={true_m:.2f} kg "
          f"(density {density:.1f} kg/m^3, hidden from controller)")
    print(f"ESTIMATED mass = {est[-1]:.2f} kg  (error {abs(est[-1]-true_m):.2f} kg, "
          f"{100*abs(est[-1]-true_m)/true_m:.1f}%)")
    print(f"per-drone share est (kg): {np.round(ctrl.share_estimates(), 2)}")
    print(f"payload lifted to z={pz[-1]:.2f} m")
    print("PASS" if abs(est[-1] - true_m) / true_m < 0.1 and pz[-1] > 1.5 else "CHECK")

    if args.graphs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OUT = os.path.join(ROOT, "results", "figures")
        os.makedirs(OUT, exist_ok=True)

        # transport1: mass estimate convergence + payload height
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts, est, color="#2c7fb8", label="estimated mass")
        ax.axhline(true_m, ls="--", color="k", label=f"true mass ({true_m:.1f} kg)")
        ax.axvline(T_DESCEND, ls=":", color="gray", label="grip (suction on)")
        ax.set(title=f"Online mass estimation — unknown '{shape}' payload",
               xlabel="time (s)", ylabel="payload mass (kg)")
        ax2 = ax.twinx()
        ax2.plot(ts, pz, color="#d95f0e", alpha=.6)
        ax2.set_ylabel("payload height (m)", color="#d95f0e")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "transport1.png"), dpi=130)
        plt.close(fig)

        # transport2: estimation accuracy across all tetromino shapes
        fig, ax = plt.subplots(figsize=(7, 4))
        shapes = list("IOTLSZ")
        trues, ests = [], []
        for s in shapes:
            dens = float(rng.uniform(8, 22))
            tm, _, e, _, _, _ = run(s, dens)
            trues.append(tm)
            ests.append(e[-1])
        x = np.arange(len(shapes))
        ax.bar(x - 0.2, trues, 0.4, label="true mass", color="#888")
        ax.bar(x + 0.2, ests, 0.4, label="estimated", color="#2c7fb8")
        ax.set(title="Mass estimation across unknown tetromino shapes",
               xticks=x, xticklabels=shapes, xlabel="tetromino", ylabel="mass (kg)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "transport2.png"), dpi=130)
        plt.close(fig)
        print(f"wrote {OUT}/transport1.png and transport2.png")


if __name__ == "__main__":
    main()
