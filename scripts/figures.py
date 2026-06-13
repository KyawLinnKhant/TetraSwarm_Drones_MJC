"""
Generate research figures (matplotlib) for the TetraSwarm paper / repo.

    python scripts/figures.py        # writes results/figures/*.png

Figures:
  1. Formation convergence + collision-free fly-in (circle/star/square/heart)
  2. Inter-drone minimum spacing vs the 1.5 m safety bound
  3. Turn-to-fit: payload yaw & success vs doorway width
  4. Maze transport: payload tilt & forward speed along the route (0 wall hits)
  5. Payload weight & required drones vs tile size (the sizing model)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.scene_builder import (build_scene, build_squeeze_scene, build_maze_scene,
                                plan_transport, payload_extent)
from control.pd_controller import SwarmPD
from llm import formations

OUT = os.path.join(ROOT, "results", "figures")
FORMS = ["circle", "star", "square", "heart"]


def _formation_run(name, n=12, seconds=6.0):
    xml = build_scene(n_drones=n, with_payload=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    ctrl = SwarmPD(m, n, kp=6, kd=4, fmax=12)
    tg = formations.make(name, n)
    c = tg[:, :2].mean(0)
    for i in range(n):
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
        d.qpos[qa:qa + 2] = c + 0.4 * (tg[i, :2] - c)
        d.qpos[qa + 2] = tg[i, 2]
    mujoco.mj_forward(m, d)
    ts, errs, seps, contacts = [], [], [], 0
    for _ in range(int(seconds / m.opt.timestep)):
        ctrl.apply(d, tg)
        mujoco.mj_step(m, d)
        contacts += d.ncon
        p = ctrl.positions(d)
        ts.append(d.time)
        errs.append(np.linalg.norm(p - tg, axis=1).mean())
        dd = np.linalg.norm(p[:, None, :2] - p[None, :, :2], axis=-1)
        np.fill_diagonal(dd, np.inf)
        seps.append(dd.min())
    return np.array(ts), np.array(errs), np.array(seps), contacts


def fig_formation():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name in FORMS:
        ts, errs, seps, contacts = _formation_run(name)
        a1.plot(ts, errs, label=f"{name} ({contacts} collisions)")
        a2.plot(ts, seps)
    a1.set(title="Formation convergence (collision-free fly-in)",
           xlabel="time (s)", ylabel="mean position error (m)")
    a1.legend(fontsize=8)
    a1.grid(alpha=.3)
    a2.axhline(1.5, ls="--", color="r", label="1.5 m safety bound")
    a2.set(title="Minimum inter-drone spacing", xlabel="time (s)",
           ylabel="min spacing (m)", ylim=(0, None))
    a2.legend(fontsize=8)
    a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_formations.png"), dpi=130)
    plt.close(fig)


def fig_squeeze():
    from scripts.demo_transport import make_transport_stepper, T_LIFT, payload_tilt_deg
    HALF_PI = np.pi / 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    # (a) yaw + x-position over time for a successful turn-through
    plan = plan_transport(shape="Z")
    xml, info = build_squeeze_scene(plan=plan, gap_w=3.5)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "payload")
    start, goal = np.array(info["start"]), np.array(info["goal"])
    ex, _ = payload_extent(info)
    path = lambda a: start + a * (goal - start)
    yaw_of = lambda a: HALF_PI if abs(path(a)[1] - info["wall_y"]) < ex + 3.0 else 0.0
    step, st = make_transport_stepper(m, d, info, goal, 2.0, path=path,
                                      yaw_of=yaw_of, yaw_rate=1.6)
    ts, yaws, ys = [], [], []
    for _ in range(int(26 / m.opt.timestep)):
        step()
        ts.append(d.time)
        yaws.append(np.degrees(st["yaw"]))
        ys.append(d.xpos[pid][1])
    a1.plot(ts, yaws, label="payload yaw (deg)")
    a1.plot(ts, ys, label="payload y-position (m)")
    a1.axhline(90, ls=":", color="gray")
    a1.axhline(info["wall_y"], ls="--", color="r", label="doorway")
    a1.set(title="Turn-to-fit: rotate 90° at the door", xlabel="time (s)")
    a1.legend(fontsize=8)
    a1.grid(alpha=.3)

    # (b) success vs gap width
    gaps = [2.0, 2.5, 3.0, 3.3, 3.6, 4.0, 4.5, 5.0]
    oks = []
    for g in gaps:
        xml, info = build_squeeze_scene(plan=plan, gap_w=g)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "payload")
        start, goal = np.array(info["start"]), np.array(info["goal"])
        ex, _ = payload_extent(info)
        path = lambda a: start + a * (goal - start)
        yaw_of = lambda a, ex=ex, info=info: (np.pi / 2
                 if abs(path(a)[1] - info["wall_y"]) < ex + 3.0 else 0.0)
        step, st = make_transport_stepper(m, d, info, goal, 2.0, path=path,
                                          yaw_of=yaw_of, yaw_rate=1.6)
        for _ in range(int(26 / m.opt.timestep)):
            step()
        oks.append(1 if np.linalg.norm(d.xpos[pid][:2] - goal) < 0.6 else 0)
    colors = ["#2a9d4a" if o else "#c0392b" for o in oks]
    a2.bar([str(g) for g in gaps], [1] * len(gaps), color=colors)
    a2.axvline(1.5, color="k", ls="--")
    a2.set(title="Delivered vs doorway width (Z narrow side = 3.0 m)",
           xlabel="doorway gap (m)", yticks=[])
    a2.text(0.4, 0.5, "blocked", color="#c0392b", rotation=90, va="center")
    a2.text(5.0, 0.5, "fits (turned)", color="#2a9d4a", rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_turn_to_fit.png"), dpi=130)
    plt.close(fig)


def fig_maze():
    from scripts.demo_maze import arc_tools
    from scripts.demo_transport import make_transport_stepper, payload_tilt_deg
    plan = plan_transport(shape="Z")
    xml, info = build_maze_scene(plan=plan)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "payload")
    pgeoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == pid}
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    path, heading = arc_tools(info["route"])
    step, st = make_transport_stepper(m, d, info, info["goal"], 2.0, path=path,
                                      yaw_of=lambda a: heading(a), yaw_rate=1.4,
                                      hold_until=14.0)
    progs, tilts, speeds, wall_hits = [], [], [], 0
    prev = d.xpos[pid][:2].copy()
    for _ in range(int(60 / m.opt.timestep)):
        step()
        progs.append(st["prog"])
        tilts.append(payload_tilt_deg(d, pid))
        cur = d.xpos[pid][:2].copy()
        speeds.append(np.linalg.norm(cur - prev) / m.opt.timestep)
        prev = cur
        for ci in range(d.ncon):
            c = d.contact[ci]
            other = c.geom2 if c.geom1 in pgeoms else c.geom1
            if ((c.geom1 in pgeoms) ^ (c.geom2 in pgeoms)) and other != ground:
                wall_hits += 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(progs, tilts)
    a1.axhline(15, ls="--", color="r", label="15° limit")
    a1.set(title=f"Maze transport: payload tilt (wall hits = {wall_hits})",
           xlabel="route progress", ylabel="tilt (deg)")
    a1.legend(fontsize=8)
    a1.grid(alpha=.3)
    a2.plot(progs, speeds)
    a2.set(title="Forward speed (dips = slowing to turn at corners)",
           xlabel="route progress", ylabel="payload speed (m/s)")
    a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_maze.png"), dpi=130)
    plt.close(fig)


def fig_sizing():
    edges = np.linspace(0.8, 2.4, 9)
    masses = [plan_transport(tile_edge=e)["payload_mass"] for e in edges]
    drones = [plan_transport(tile_edge=e)["n_drones"] for e in edges]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(edges, masses, "o-", color="#2c7fb8", label="payload mass (kg)")
    ax.set(xlabel="tile edge (m)", ylabel="payload mass (kg)",
           title="Cleveland-Z sizing model")
    ax2 = ax.twinx()
    ax2.plot(edges, drones, "s--", color="#d95f0e", label="drones needed")
    ax2.set_ylabel("drones needed")
    ax.grid(alpha=.3)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_sizing.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig_formation, fig_squeeze, fig_maze, fig_sizing):
        print("rendering", fn.__name__, "...")
        fn()
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
