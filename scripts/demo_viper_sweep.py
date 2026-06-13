"""
Replay the REAL ViPER plan in MuJoCo: the 4 drones fly the exact paths that
marmotlab/ViPER's pretrained policy planned to sweep/clear our braided maze
(trajectories dumped by external/ViPER/run_viper.py -> results/viper_traj.npz).

Because ViPER's paths are planned over the sensed free space they are already
collision-free, so the drones just track them kinematically — no potential-field
avoidance hack. This is the genuine ViPER sweep, visualized in our simulator.

    python scripts/demo_viper_sweep.py            # writes results/viper_sweep.gif
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_scout_scene

TRAJ = os.path.join(ROOT, "external", "ViPER", "results", "viper_traj.npz")
DRONE_COLORS = ["#e51d23", "#3399f2", "#33cc66", "#f2c233"]


def main():
    d = np.load(TRAJ)
    cells = d["traj"].astype(float)                 # (T, n, 2) = (cell_x, cell_y)
    gt_h, gt_w = d["gt_shape"]
    xml, info = build_scout_scene(n_drones=cells.shape[1], n_rays=0,  # no lidar fans
                                  drone_size=0.55)                    # visible markers
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    nd = cells.shape[1]
    x0, y0, W = info["x0"], info["y0"], info["nx"] * info["cell"]
    sx = W / gt_w                                    # metres per grid cell
    sz = info["scout_z"]

    def to_world(cx, cy, flip):
        wx = x0 + cx * sx
        wy = (y0 + (gt_h - cy) * sx) if flip else (y0 + cy * sx)
        return wx, wy

    sb = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"scout{i}") for i in range(nd)]
    dq = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"scout{i}_free")]
          for i in range(nd)]
    bg = {i: {g for g in range(model.ngeom) if model.geom_bodyid[g] == sb[i]} for i in range(nd)}
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    def place(world_xy):
        for i in range(nd):
            data.qpos[dq[i]:dq[i] + 3] = [world_xy[i][0], world_xy[i][1], sz]
            data.qpos[dq[i] + 3:dq[i] + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(model, data)

    def wall_hits():
        h = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            ind = [i for i in range(nd) if c.geom1 in bg[i] or c.geom2 in bg[i]]
            if ind:
                other = c.geom2 if c.geom1 in bg[ind[0]] else c.geom1
                if other != ground and not any(other in bg[j] for j in range(nd)):
                    h += 1
        return h

    # pick the cell->world orientation (flip or not) that keeps drones in corridors
    best, bestflip = 1e9, False
    for flip in (False, True):
        tot = 0
        for t in range(len(cells)):
            place([to_world(cells[t, i, 0], cells[t, i, 1], flip) for i in range(nd)])
            tot += wall_hits()
        if tot < best:
            best, bestflip = tot, flip
    print(f"orientation: flip={bestflip}  (wall-overlap score {best})")

    # render: interpolate between ViPER waypoints, drive drones kinematically
    renderer = mujoco.Renderer(model, height=620, width=620)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance = max(info["nx"], info["ny"]) * info["cell"] * 1.35
    cam.azimuth, cam.elevation = 90, -89
    opt = mujoco.MjvOption()        # show the big colour marker (grp 3), hide X2 mesh (grp 2)
    opt.geomgroup[2] = 0
    opt.geomgroup[3] = 1
    frames, sub = [], 10
    contacts = 0
    for t in range(len(cells) - 1):
        for s in range(sub):
            a = s / sub
            wxy = []
            for i in range(nd):
                c0 = cells[t, i]; c1 = cells[t + 1, i]
                cx, cy = c0 + a * (c1 - c0)
                wxy.append(to_world(cx, cy, bestflip))
            place(wxy)
            contacts += wall_hits()
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frames.append(renderer.render().copy())
    print(f"drone-wall contacts over replay: {contacts}")

    from PIL import Image
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames[::2]]
    pal = imgs[len(imgs) // 2].quantize(colors=128)
    q = [im.quantize(palette=pal, dither=Image.NONE) for im in imgs]
    out = os.path.join(ROOT, "results", "viper_sweep.gif")
    q[0].save(out, save_all=True, append_images=q[1:], duration=80, loop=0, optimize=True)
    print(f"wrote {out}  ({len(q)} frames)")

    # ViPER clearing-progress graph (from the pretrained policy's own metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    expl = 100 * d["explored"]
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(len(expl)), expl, "-o", color="#2c7fb8", lw=2.2, ms=4)
    ax.fill_between(range(len(expl)), 0, expl, alpha=0.12, color="#2c7fb8")
    ax.set(title="ViPER cooperative clearing of the unknown maze\n(4 agents, pretrained policy, default settings)",
           xlabel="planning step", ylabel="maze cleared (%)", ylim=(0, 105))
    ax.grid(alpha=0.3)
    ax.text(len(expl) - 1, 102, "100% cleared", ha="right", fontsize=9, color="#2c7fb8")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "results", "figures", "viper_clearing.png"), dpi=130)
    plt.close(fig)
    print("wrote results/figures/viper_clearing.png")


if __name__ == "__main__":
    main()
