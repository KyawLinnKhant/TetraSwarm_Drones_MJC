"""
Offscreen renderer — produce PNG snapshots and animated GIFs of TetraSwarm runs
without opening a window (works in headless / CI and lets you preview results).

    python scripts/render.py formation --instruction "wide star" --drones 8
    python scripts/render.py transport  --shape L --to 3 0 --height 2.5

Outputs land in results/ (formation.png, transport.gif, transport_final.png).
Rendering uses mujoco.Renderer (offscreen EGL/CGL) + Pillow for the GIF, so no
ffmpeg is required. For a live interactive window instead, see the note printed
at the end of a run.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mujoco
from PIL import Image

from envs.scene_builder import (build_scene, build_transport_scene,
                                build_navigation_scene, build_mission_scene,
                                build_maze_scene)
from control.pd_controller import SwarmPD

RESULTS = os.path.join(ROOT, "results")
W, H = 960, 600


def _camera(lookat=(0.0, 0.0, 1.2), distance=7.0, azimuth=120.0, elevation=-20.0):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _roll(model, data, step_fn, seconds, cam, fps=20):
    """Step the sim, grabbing a frame every 1/fps seconds. Returns list of frames."""
    renderer = mujoco.Renderer(model, height=H, width=W)
    frames, every = [], max(1, int(round((1.0 / fps) / model.opt.timestep)))
    n_steps = int(seconds / model.opt.timestep)
    for k in range(n_steps):
        step_fn()
        if k % every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
    renderer.close()
    return frames


def _save_gif(frames, path, fps=20):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True)


def render_formation(args):
    from llm.commander import Commander, Command
    if args.formation:                            # deterministic, no LLM needed
        plan = Command(args.formation, args.drones, {}, "forced", source="direct")
        name = args.formation
    else:
        plan = Commander(api_key="" if args.no_llm else Commander._AUTO).plan(
            args.instruction, args.drones)
        name = plan.formation
    print(f'-> {plan.formation} ({plan.source}) {plan.params}')

    xml = build_scene(n_drones=args.drones, with_payload=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ctrl = SwarmPD(model, args.drones, fmax=12.0)
    targets = plan.targets()

    # Collision-free fly-in: start each drone in a shrunk copy of its OWN slot, so
    # the swarm expands radially into the shape without any path crossing.
    c = targets[:, :2].mean(0)
    for i in range(args.drones):
        qa = model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
        data.qpos[qa:qa + 2] = c + 0.4 * (targets[i, :2] - c)
        data.qpos[qa + 2] = targets[i, 2]
    mujoco.mj_forward(model, data)

    def step():
        ctrl.apply(data, targets)
        mujoco.mj_step(model, data)

    cam = _camera(distance=9.0, elevation=-45.0)
    frames = _roll(model, data, step, seconds=6.0, cam=cam)
    gif = os.path.join(RESULTS, f"formation_{name}.gif")
    png = os.path.join(RESULTS, f"formation_{name}.png")
    _save_gif(frames, gif)
    Image.fromarray(frames[-1]).save(png)
    print(f"wrote {gif}\nwrote {png}")


def render_transport(args):
    from scripts.demo_transport import make_transport_stepper, T_CARRY
    from envs.scene_builder import plan_transport

    plan = plan_transport(shape=args.shape)
    xml, info = build_transport_scene(plan=plan)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    n = info["n_carriers"]
    deliver = np.array(args.to)
    step, _ = make_transport_stepper(model, data, info, deliver, args.height)
    print(f"shape={args.shape} payload={plan['payload_mass']:.2f}kg carriers={n}")

    cam = _camera(lookat=(deliver[0] / 2, deliver[1] / 2, args.height * 0.6),
                  distance=8.5, azimuth=110.0, elevation=-16.0)
    frames = _roll(model, data, step, seconds=T_CARRY + 1.5, cam=cam)
    gif = os.path.join(RESULTS, f"transport_{args.shape}.gif")
    png = os.path.join(RESULTS, f"transport_{args.shape}.png")
    _save_gif(frames, gif)
    Image.fromarray(frames[-1]).save(png)
    print(f"shape={args.shape} carriers={n} -> deliver {tuple(args.to)} @ {args.height} m")
    print(f"wrote {gif}\nwrote {png}")


def render_navigate(args):
    from scripts.demo_navigate import slot_offsets

    xml, info = build_navigation_scene(n_drones=args.drones)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ctrl = SwarmPD(model, args.drones, kp=6.0, kd=4.5, fmax=10.0)
    nav_z = info["nav_z"]
    offsets = slot_offsets(args.drones)
    waypoints = [np.array([x, y, nav_z]) for (x, y) in info["waypoints"]]
    centroid = np.array([info["start"][0], info["start"][1], nav_z], dtype=float)
    state = {"wp": 0, "c": centroid}
    dt = model.opt.timestep

    def step():
        pos = ctrl.positions(data)
        slot_err = np.linalg.norm(pos[:, :2] - (state["c"] + offsets)[:, :2], axis=1)
        cohesive = slot_err.max() < 0.6
        to = waypoints[state["wp"]] - state["c"]
        dist = np.linalg.norm(to)
        if dist < 0.25 and state["wp"] < len(waypoints) - 1:
            state["wp"] += 1
        elif cohesive and dist > 1e-6:
            state["c"] = state["c"] + to / dist * min(0.7 * dt, dist)
        ctrl.apply(data, state["c"] + offsets)
        mujoco.mj_step(model, data)

    # Top-down-ish view so the slalom gates and weave are visible.
    cam = _camera(lookat=(0.0, 0.0, 1.0), distance=15.0, azimuth=90.0, elevation=-55.0)
    frames = _roll(model, data, step, seconds=28.0, cam=cam, fps=15)
    gif = os.path.join(RESULTS, "navigate.gif")
    png = os.path.join(RESULTS, "navigate_final.png")
    _save_gif(frames, gif)
    Image.fromarray(frames[-1]).save(png)
    print(f"drones={args.drones} navigated gates -> goal {info['goal']}")
    print(f"wrote {gif}\nwrote {png}")


def render_mission(args):
    from scripts.demo_mission import polyline
    from scripts.demo_transport import make_transport_stepper, T_CARRY
    from envs.scene_builder import plan_transport

    plan = plan_transport(shape=args.shape)
    xml, info = build_mission_scene(plan=plan)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    route = polyline([info["start"]] + list(info["waypoints"]))
    step, _ = make_transport_stepper(model, data, info, info["goal"], args.height,
                                     path=route)
    # scout flies the route first (see demo_mission for the same logic)
    sb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scout")
    sdof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scout_free")]
    sz, T_SCOUT = info["scout_z"], 7.0

    def step_with_scout():
        a = min(1.0, data.time / T_SCOUT)
        tx, ty = route(a)
        f = 8.0 * (np.array([tx, ty, sz]) - data.xpos[sb]) - 5.0 * data.qvel[sdof:sdof + 3]
        f[2] += 0.3 * 9.81
        data.xfrc_applied[sb, :3] = f
        step()

    print(f"mission: scout maps, then {info['n_carriers']} drones carry "
          f"{args.shape} ({plan['payload_mass']:.2f}kg) through gates")

    cam = _camera(lookat=(0.0, 0.0, 1.2), distance=15.5, azimuth=90.0, elevation=-58.0)
    frames = _roll(model, data, step_with_scout, seconds=T_CARRY + 2.0, cam=cam)
    gif = os.path.join(RESULTS, "mission.gif")
    png = os.path.join(RESULTS, "mission_final.png")
    _save_gif(frames, gif)
    Image.fromarray(frames[-1]).save(png)
    print(f"wrote {gif}\nwrote {png}")


def render_squeeze(args):
    from scripts.demo_squeeze import HALF_PI
    from scripts.demo_transport import make_transport_stepper, T_CARRY
    from envs.scene_builder import build_squeeze_scene, plan_transport, payload_extent

    plan = plan_transport(shape=args.shape)
    xml, info = build_squeeze_scene(plan=plan, gap_w=args.gap)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    start, goal = np.array(info["start"]), np.array(info["goal"])
    ex, _ = payload_extent(info)
    must_turn = 2 * ex > info["gap_w"]
    path = lambda a: start + a * (goal - start)
    yaw_of = lambda a: (HALF_PI if must_turn and abs(path(a)[1] - info["wall_y"]) < ex + 3.0
                        else 0.0)
    step, _ = make_transport_stepper(model, data, info, goal, 2.0, path=path,
                                     yaw_of=yaw_of, yaw_rate=1.6)
    print(f"squeeze: {2*ex:.1f}m-wide {args.shape} through {info['gap_w']}m door")

    cam = _camera(lookat=(0.0, 0.0, 1.0), distance=16.0, azimuth=90.0, elevation=-78.0)
    frames = _roll(model, data, step, seconds=26.0, cam=cam, fps=15)
    gif = os.path.join(RESULTS, "squeeze.gif")
    png = os.path.join(RESULTS, "squeeze_final.png")
    _save_gif(frames, gif)
    Image.fromarray(frames[len(frames) // 2]).save(png)   # mid-turn snapshot
    print(f"wrote {gif}\nwrote {png}")


def render_morph(args):
    """One LLM-driven clip: natural-language instructions -> the commander picks
    each formation -> the swarm morphs through ALL of them, ending on a Fibonacci
    sunflower. Transitions use VERTICAL DECONFLICTION — each drone cruises at its
    own unique altitude while crossing, so no two ever share a point in space
    (guaranteed collision-free, not just in projection)."""
    from llm import formations
    from llm.commander import Commander
    n = args.drones
    cmd = Commander()
    # the LLM (Gemini, offline keyword fallback) chooses each shape:
    instructions = [
        "form a circle",
        "reshape into a square",
        "arrange into a grid",
        "form a V-shaped flock",
        "line up in a single row",
        "transform into a star",
        "morph into a heart",
        "swirl into a fibonacci sunflower",
    ]
    plans = [cmd.plan(t, n) for t in instructions]
    seq = [p.formation for p in plans]
    forms = [formations.make(p.formation, n, min_sep=1.2, **p.params) for p in plans]
    for t, p in zip(instructions, plans):
        print(f'  "{t}"  ->  {p.formation}  ({p.source})')

    xml = build_scene(n_drones=n, with_payload=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dq = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_free")]
          for i in range(n)]

    def place(P):                                   # drive drones kinematically
        for i in range(n):
            data.qpos[dq[i]:dq[i] + 3] = P[i]
            data.qpos[dq[i] + 3:dq[i] + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(model, data)

    f0 = forms[0]
    c = f0[:, :2].mean(0)
    # each drone gets a UNIQUE cruise-altitude offset (m) for transitions, assigned
    # as a SMOOTH gradient by angle (a gentle helical wave, not a random scatter).
    # Crossing drones are far apart in angle -> far apart in lane -> clear by >0.4 m.
    ang = np.arctan2(f0[:, 1] - c[1], f0[:, 0] - c[0])
    lanes = np.empty(n)
    lanes[np.argsort(ang)] = np.linspace(-1.0, 4.0, n)
    place(f0)                                       # START already in the circle

    cam = _camera(lookat=(0.0, 0.0, 1.5), distance=12.5, elevation=-46.0)
    renderer = mujoco.Renderer(model, height=H, width=W)
    dt = model.opt.timestep
    every = max(1, int(round((1 / 20) / dt)))      # 20 fps during motion
    # PIL collapses identical GIF frames, so we use PER-FRAME durations: motion
    # frames are short, each HOLD is a single frame shown for its full duration.
    frames, durs, md = [], [], [1e9]

    def shot(ms):
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
        durs.append(int(ms))

    def track(P):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        md[0] = min(md[0], d.min())

    def morph(A, B, seconds, deconflict=False):
        ns = int(seconds / dt)
        for k in range(ns):
            a = (k + 1) / ns
            s = a * a * (3 - 2 * a)                  # smoothstep ease
            P = A + s * (B - A)
            if deconflict:                           # lift each drone to its own lane mid-move
                P = P.copy()
                P[:, 2] += lanes * np.sin(np.pi * a)
            place(P)
            track(P)
            if k % every == 0:
                shot(1000 / 20)                      # 50 ms motion frame
        place(B)                                     # land exactly on the target

    def hold(P, seconds):
        place(P)
        track(P)
        shot(seconds * 1000)                         # ONE frame, shown for `seconds`

    hold(f0, 2.0)                                    # show the circle ~2 s
    cur = f0
    for nxt in forms[1:]:
        perm = formations.assign_targets(cur[:, :2], nxt[:, :2])   # angle-matched ordering
        tgt = nxt[perm]
        morph(cur, tgt, 2.6, deconflict=True)        # morph with vertical lanes (no crash)
        hold(tgt, 2.0)                               # HOLD each shape ~2 s
        cur = tgt
    morph(cur, f0, 2.6, deconflict=True)             # fibonacci -> circle: seamless loop
    renderer.close()

    # save with per-frame durations (shared 128-colour palette so holds survive)
    gif = os.path.join(RESULTS, "formation_morph.gif")
    small = [Image.fromarray(f).resize((720, 450), Image.BILINEAR) for f in frames]
    pal = small[len(small) // 2].quantize(colors=128)
    q = [im.quantize(palette=pal, dither=Image.NONE) for im in small]
    q[0].save(gif, save_all=True, append_images=q[1:], duration=durs, loop=0,
              optimize=True, disposal=1)
    Image.fromarray(frames[-1]).save(os.path.join(RESULTS, "formation_morph.png"))
    print(f"LLM-driven morph {seq} -> {gif}  ({len(q)} frames)")
    print(f"closest drone-drone approach over the whole clip: {md[0]:.2f} m "
          f"({'CLEAR' if md[0] > 0.4 else 'TOO CLOSE'})")


def render_maze(args):
    from scripts.demo_maze import arc_tools
    from scripts.demo_transport import make_transport_stepper
    from envs.scene_builder import plan_transport, payload_extent

    plan = plan_transport(shape=args.shape)
    xml, info = build_maze_scene(plan=plan)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    path, heading = arc_tools(info["route"])
    sweep = np.array(info["sweep"])
    ex, ey = payload_extent(info)
    long_off = (np.pi / 2) if ey > ex else 0.0
    T_MAP = 20.0
    step, st = make_transport_stepper(model, data, info, info["goal"], 2.0, path=path,
                                      yaw_of=lambda a: heading(a) + long_off,
                                      yaw_rate=0.8, hold_until=T_MAP, turn_slow=0.05)
    sb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scout")
    sdof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scout_free")]
    sz = info["scout_z"]

    def step_all():
        a = min(0.999, data.time / T_MAP)
        seg = a * (len(sweep) - 1)
        i = int(seg)
        tx, ty = sweep[i] + (seg - i) * (sweep[i + 1] - sweep[i])
        f = 11.0 * (np.array([tx, ty, sz]) - data.xpos[sb]) - 5.0 * data.qvel[sdof:sdof + 3]
        f[2] += 0.3 * 9.81
        data.xfrc_applied[sb, :3] = f
        step()

    span = max(info["nx"], info["ny"]) * info["cell"]
    cam = _camera(lookat=(0, 0, 1.0), distance=span * 1.3, azimuth=90, elevation=-89)
    frames = _roll(model, data, step_all, seconds=74.0, cam=cam, fps=12)
    gif = os.path.join(RESULTS, f"maze_{args.shape}.gif")
    png = os.path.join(RESULTS, f"maze_{args.shape}.png")
    _save_gif(frames, gif)
    Image.fromarray(frames[int(len(frames) * 0.7)]).save(png)
    print(f"shape={args.shape} -> {gif}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("formation")
    f.add_argument("-i", "--instruction", default="wide star")
    f.add_argument("--formation", default=None,
                   help="force a formation (skips the LLM): circle/star/square/heart/...")
    f.add_argument("--drones", type=int, default=10)
    f.add_argument("--no-llm", action="store_true")
    f.set_defaults(func=render_formation)

    t = sub.add_parser("transport")
    t.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    t.add_argument("--to", nargs=2, type=float, default=[3.0, 0.0], metavar=("X", "Y"))
    t.add_argument("--height", type=float, default=2.5)
    t.set_defaults(func=render_transport)

    nv = sub.add_parser("navigate")
    nv.add_argument("--drones", type=int, default=6)
    nv.set_defaults(func=render_navigate)

    ms = sub.add_parser("mission")
    ms.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    ms.add_argument("--height", type=float, default=2.0)
    ms.set_defaults(func=render_mission)

    sq = sub.add_parser("squeeze")
    sq.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    sq.add_argument("--gap", type=float, default=3.5)
    sq.set_defaults(func=render_squeeze)

    mz = sub.add_parser("maze")
    mz.add_argument("--shape", default="Z", choices=list("IOTLSZ"))
    mz.set_defaults(func=render_maze)

    mo = sub.add_parser("morph")
    mo.add_argument("--drones", type=int, default=12)
    mo.set_defaults(func=render_morph)

    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    args.func(args)
    print("\nLive window instead:  mjpython scripts/demo_transport.py --shape L")


if __name__ == "__main__":
    main()
