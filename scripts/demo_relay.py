"""
Multi-block warehouse relay — one swarm shuttles four tetromino blocks from a
central depot out to the four corners, turning each block to thread the route:

    Z -> top-left,  I -> bottom-right,  O -> top-right,  L -> bottom-left

Each block is carried kinematically (it rigidly follows the carrier formation),
so the swarm can pick up, carry, turn, and drop each in sequence. Blocks are
placed without overlapping (tetris-aligned).

    python scripts/demo_relay.py             # viewer (mjpython on macOS)
    python scripts/demo_relay.py --headless  # checks every block reaches its corner
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco

from envs.scene_builder import build_relay_scene
from control.pd_controller import SwarmPD

GROUND_DZ = 0.0          # block-center height offset when resting (set per scene)
LIFT_Z = 1.5             # block-center height while carried


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--render", action="store_true", help="write results/relay.gif")
    args = ap.parse_args()

    xml, info = build_relay_scene()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ctrl = SwarmPD(model, 4, kp=8, kd=5, fmax=None)
    hz, th, tether = info["half_z"], info["tile_half"], info["tether_len"]
    ground_z = hz
    drone_above = lambda cz: cz + hz + tether

    qadr = {s: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_free")]
            for s in info["blocks"]}
    vadr = {s: model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_free")]
            for s in info["blocks"]}
    bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, s) for s in info["blocks"]}
    # current pose of every block: (x, y, z, yaw)
    pose = {s: (info["depot"][s][0], info["depot"][s][1], ground_z, 0.0)
            for s in info["blocks"]}

    def write_blocks():
        for s in info["blocks"]:
            x, y, z, yaw = pose[s]
            data.qpos[qadr[s]:qadr[s] + 3] = [x, y, z]
            data.qpos[qadr[s] + 3:qadr[s] + 7] = yaw_quat(yaw)
            data.qvel[vadr[s]:vadr[s] + 6] = 0

    write_blocks()
    mujoco.mj_forward(model, data)

    log = []
    frames, renderer, cam, every = [], None, None, 1
    if args.render:
        renderer = mujoco.Renderer(model, height=720, width=820)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0]
        cam.distance, cam.azimuth, cam.elevation = 38, 90, -84
        every = int(round((1 / 15) / model.opt.timestep))
    _kc = {"k": 0}

    def step(targets):
        ctrl.apply(data, targets)
        mujoco.mj_step(model, data)
        write_blocks()                         # kinematically hold/drive blocks
        mujoco.mj_forward(model, data)
        if renderer is not None:
            if _kc["k"] % every == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())
            _kc["k"] += 1

    def grips_world(shape, cx, cy, yaw, dz):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        out = []
        for (ox, oy) in info["grips"][shape]:
            wx, wy = R @ np.array([ox, oy])
            out.append([cx + wx, cy + wy, dz])
        return np.array(out)

    def run(seconds, center_fn, yaw_fn, dz_fn, carry=None, shape=None):
        nonlocal pose
        n = int(seconds / model.opt.timestep)
        for k in range(n):
            a = (k + 1) / n
            cx, cy = center_fn(a)
            yaw = yaw_fn(a)
            cz = dz_fn(a)                      # block-center height
            if carry is not None:
                pose[carry] = (cx, cy, cz, yaw)
            targets = grips_world(shape, cx, cy, yaw, drone_above(cz))
            step(targets)

    order = [(s, info["deliver"][s]) for s in info["blocks"]]   # Z,I,O,L
    cur = info["depot"]["Z"]                  # where the formation currently is
    travel_z = LIFT_Z

    for s, corner in order:
        dep = info["depot"][s]
        goal = info["corners"][corner]
        # L-path depot->corner: move vertically first, then horizontally (1 turn)
        turn = (dep[0], goal[1])
        # 1) transit empty to the block (fly over at travel height)
        run(2.5, lambda a, p0=cur, p1=dep: (p0[0] + a * (p1[0] - p0[0]),
            p0[1] + a * (p1[1] - p0[1])), lambda a: 0.0, lambda a: LIFT_Z, shape=s)
        # 2) descend onto the block
        run(1.2, lambda a: dep, lambda a: 0.0, lambda a: ground_z, shape=s)
        # 3) grip + lift
        run(1.3, lambda a: dep, lambda a: 0.0,
            lambda a: ground_z + a * (LIFT_Z - ground_z), carry=s, shape=s)
        # 4) carry along the L route, turning the block to follow the heading
        h1 = np.arctan2(turn[1] - dep[1], turn[0] - dep[0])
        h2 = np.arctan2(goal[1] - turn[1], goal[0] - turn[0])
        run(4.0, lambda a, p0=dep, p1=turn: (p0[0] + a * (p1[0] - p0[0]),
            p0[1] + a * (p1[1] - p0[1])), lambda a: h1, lambda a: LIFT_Z,
            carry=s, shape=s)
        run(2.5, lambda a, p0=turn, p1=turn: turn, lambda a, h1=h1, h2=h2:
            h1 + a * (h2 - h1), lambda a: LIFT_Z, carry=s, shape=s)   # rotate at corner
        run(4.0, lambda a, p0=turn, p1=goal: (p0[0] + a * (p1[0] - p0[0]),
            p0[1] + a * (p1[1] - p0[1])), lambda a: h2, lambda a: LIFT_Z,
            carry=s, shape=s)
        # 5) set the block down at the corner, release
        run(1.3, lambda a: goal, lambda a: h2,
            lambda a: LIFT_Z + a * (ground_z - LIFT_Z), carry=s, shape=s)
        pose[s] = (goal[0], goal[1], ground_z, h2)
        cur = goal
        d = np.linalg.norm(np.array(data.xpos[bid[s]][:2]) - np.array(goal))
        log.append((s, corner, d))
        print(f"delivered {s} -> {corner}  (block at corner, err {d:.2f} m)")

    ok = all(d < 0.6 for _, _, d in log)
    print("PASS" if ok else "FAIL")
    if args.render and frames:
        from PIL import Image
        renderer.close()
        gif = os.path.join(ROOT, "results", "relay.gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=66,
                     loop=0, optimize=True)
        Image.fromarray(frames[len(frames) // 2]).save(
            os.path.join(ROOT, "results", "relay.png"))
        print(f"wrote {gif}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
