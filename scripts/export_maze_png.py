"""
Export the TetraSwarm braided maze as a ViPER-format occupancy PNG so the REAL
marmotlab/ViPER policy can plan on OUR maze.

ViPER map convention (see external/ViPER/env.py import_ground_truth):
    127 = obstacle,  195 = free space,  208 = agent start cell.

The maze topology is rasterized at ~PX pixels across; ViPER reads each pixel as
CELL_SIZE (0.4 m) so the map is nominally scaled up — this is only to give ViPER's
default 20 m sensor a meaningfully large map to explore (the topology is identical).

    python scripts/export_maze_png.py            # -> external/ViPER/maps_spec/map.png
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco
from PIL import Image

from envs.scene_builder import build_scout_scene


def main(px=480, out=None):
    xml, info = build_scout_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    nx, ny, cell, x0, y0 = info["nx"], info["ny"], info["cell"], info["x0"], info["y0"]
    W = nx * cell                                   # maze span (square)
    scale = px / W                                  # pixels per metre
    img = np.full((px, px), 195, np.uint8)          # start all-free inside

    def m2p(x, y):                                  # world metre -> pixel (row, col)
        c = int((x - x0) * scale)
        r = int((y - y0) * scale)
        return np.clip(r, 0, px - 1), np.clip(c, 0, px - 1)

    # rasterize every wall box (worldbody box geoms; skip ground plane + drones)
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    for g in range(model.ngeom):
        if g == ground or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        if model.geom_bodyid[g] != 0:               # only world-attached walls
            continue
        cx, cy = data.geom_xpos[g][:2]
        hx, hy = model.geom_size[g][:2]
        r0, c0 = m2p(cx - hx, cy - hy)
        r1, c1 = m2p(cx + hx, cy + hy)
        img[min(r0, r1):max(r0, r1) + 1, min(c0, c1):max(c0, c1) + 1] = 127

    # agent start cell = the depot centre (a small 208 patch in free space)
    sr, sc = m2p(*info["center"])
    img[img == 127] = 127
    k = max(2, int(0.6 * scale))
    patch = img[sr - k:sr + k, sc - k:sc + k]
    patch[patch == 195] = 208                       # only mark free cells as start

    out = out or os.path.join(ROOT, "external", "ViPER", "maps_spec", "map.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Image.fromarray(img, mode="L").save(out)
    free = int((img == 195).sum()); obs = int((img == 127).sum()); st = int((img == 208).sum())
    print(f"wrote {out}  ({px}x{px}; free={free} obstacle={obs} start={st})")


if __name__ == "__main__":
    main()
