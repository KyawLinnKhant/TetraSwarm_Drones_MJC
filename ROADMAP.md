# TetraSwarm — Architecture & Roadmap

An LLM-commanded MuJoCo drone swarm that drives N quadrotors into formations and
(in progress) cooperatively transports rigid **tetromino** payloads.

The system is built in layers. Each layer has a single responsibility and a
clean interface to the next, so the LLM only ever *selects and parameterizes* —
it never emits raw coordinates, and the simulator only ever sees a validated
command.

Legend: ✅ built · ⬜ planned

| # | Layer | Responsibility | Files | Status |
|---|-------|----------------|-------|--------|
| **0** | Simulation environment | MJCF scene generator: N drones (force-controlled boxes w/ freejoints) + rigid tetromino payload (welded unit cubes) + ground plane | `envs/scene_builder.py` | ✅ |
| **0** | Low-level control | Force-based PD position controller — writes a 6D wrench to `data.xfrc_applied` (gravity comp + keep-level damping) | `control/pd_controller.py` | ✅ |
| **0** | Formation geometry | Deterministic library: formation name + drone count → `(n,3)` world targets. circle / star / line / vee / grid | `llm/formations.py` | ✅ |
| **1** | LLM commander | Natural-language instruction → Gemini selects & parameterizes a formation → validated `Command` (registry-checked, keyword fallback when offline) | `llm/commander.py`, `scripts/demo_commander.py` | ✅ |
| **2** | Cooperative payload transport | **Auto-sized** job: `plan_transport` computes payload mass (cube size×density) + the **necessary** drone count (lift capacity + margin, ≥1/cube). Each carrier hangs a **suction-cup tether**; runs approach→descend→**suction on** (`connect`, `active=false`→`eq_active`)→lift→carry. Supports the **Cleveland Z** | `envs/scene_builder.py` (`plan_transport`, `build_transport_scene`, `_transport_pieces`), `scripts/demo_transport.py` | ✅ |
| **3** | RL formation-keeping policy | Learned controller to replace/augment the PD baseline under disturbance & payload load | — | ⬜ |
| **4** | Real drone model (Skydio X2) | X2 mesh used for all drones (visual mesh + invisible mass box); force-control pipeline unchanged. *Not yet rotor-level dynamics — props don't spin* | `envs/scene_builder.py` (`_x2_asset_block`, `_drone_body`) | ◑ |
| **5** | Navigation / waypoints | Swarm weaves through a walled slalom (two offset gates) to a goal: moving formation centroid follows deep-gap waypoints with a cohesion gate | `envs/scene_builder.py` (`build_navigation_scene`, `_arena_layout`), `scripts/demo_navigate.py` | ✅ |
| **2+5** | **Mission** (pick-up + deliver) | Grip a payload at the start, then **navigate it through the maze** to the goal — transport + slalom combined | `envs/scene_builder.py` (`build_mission_scene`), `scripts/demo_mission.py` | ✅ |

> Layer labels 0–1 come from the code comments. Layers 2–5 are derived from the
> "later"/TODO notes in those same files — the intended direction, not yet
> labeled in code.

## Data flow (current)

```
"spread into a wide star"
        │
        ▼
  Commander.plan()  ── Gemini (gemini-2.5-flash, structured JSON) ──▶ {formation, params}
        │                                   │ validate vs formations.REGISTRY
        ▼                                   │ keyword fallback if offline
  formations.make(name, n, **params) ──▶ targets (n,3)
        │
        ▼
  SwarmPD.apply(data, targets) ──▶ data.xfrc_applied ──▶ mujoco.mj_step
```

## Running the demos

```bash
# Layer 1 — natural language → formation (viewer)
python scripts/demo_commander.py -i "spread into a wide star"

# Headless convergence check (CI-friendly)
python scripts/demo_commander.py -i "tight defensive ring" --headless

# Offline keyword fallback (no API call)
python scripts/demo_commander.py -i "flying wedge" --no-llm

# Layer 0 — formation library only, no LLM
python scripts/demo_formation.py --formation star --headless

# Layer 2 — auto-sized suction-cup transport (Cleveland Z = 4.61 kg, 4 drones)
python scripts/demo_transport.py --shape Z --headless

# Layer 5 — navigate a walled slalom to the goal
python scripts/demo_navigate.py --drones 6 --headless

# Mission — pick up the payload and carry it through the maze to the goal
python scripts/demo_mission.py --shape Z --headless

# Render any run to results/*.gif (offscreen, no window)
python scripts/render.py transport --shape Z
python scripts/render.py navigate --drones 6
python scripts/render.py mission --shape Z
```

## Environment / Gemini notes

- SDK: **`google-genai`** (`from google import genai`) — *not* the old
  `google-generativeai`.
- Working free-tier model: **`gemini-2.5-flash`** (also `gemini-flash-latest`).
  `gemini-2.0-flash-lite` hit 429 quota; `gemini-1.5-flash` is retired (404).
- The API key lives in `.env` as `GEMINI_API_KEY` (gitignored; see
  `.env.example`). The commander calls `load_dotenv(override=True)` so the
  project `.env` wins over any stale `GEMINI_API_KEY` already in the shell.

## External references to use

- **Scene editor** (build/edit MJCF scenes visually): https://github.com/markusgrotz/mujoco-scene-editor
- **Skydio X2 drone XML** (real rotor-level drone model for Layer 4, replaces the
  flat box drones): https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2
  — note: a copy already sits in `assets/menagerie_tmp/skydio_x2/`.
- **MuJoCo drones gym** (RL env reference for drone control/navigation):
  https://github.com/tau-intelligence/MuJoCo-drones-gym
- **MuJoCo scenes** (reusable scene/world assets, walls, maps):
  https://github.com/kscalelabs/mujoco-scenes
- Idea: build a navigation arena inspired by "Viper" maps (maze/corridor layout).

## Next step

Layers 0–2, 5 work end-to-end; Layer 4 (drone model) is visual-only. Open
directions:
- **Layer 3 — RL formation-keeping policy** (the big one).
- **Wire commander → transport/navigation**: let Gemini pick payload shape,
  delivery point, or nav goal from natural language.
- **Richer maps**: build a "Viper"-style arena (see references) via the scene
  editor / kscale mujoco-scenes.

Known simplifications to revisit:
- Drones are **non-colliding** force-controlled point-followers (collision was
  removed to kill contact blow-ups). Reactive obstacle/inter-drone avoidance is
  future work.
- Carriers are **rigidly** linked (`connect`), not on slack cables; a spatial
  tendon would let the load swing.
- X2 is a **visual mesh only** — no rotor-level thrust dynamics yet.
