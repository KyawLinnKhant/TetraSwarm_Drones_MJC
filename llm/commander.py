"""
TetraSwarm — Layer 1: the LLM commander.

Turns a free-form natural-language instruction ("spread the drones into a wide
star") into a *validated* formation command that the rest of the pipeline can
execute deterministically:

    {"formation": "star", "params": {"outer": 3.0, "inner": 1.2}, "reasoning": ...}

Design choices that keep this reliable:
  - The LLM only ever *selects and parameterizes* a formation from
    ``formations.REGISTRY``. It never emits raw coordinates, so a bad answer
    can't crash the simulator — at worst we ignore an unknown param.
  - We constrain the model with a JSON response schema (structured output) and
    then re-validate in Python: the formation must be in the registry and we
    drop any param the chosen formation's signature doesn't accept.
  - If there's no API key, or the call fails/quota-limits, we fall back to a
    deterministic keyword matcher so the demo always runs.

Uses the modern ``google-genai`` SDK (``from google import genai``).
"""
from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass, field

from llm import formations

# Models that have shown free-tier quota; tried in order on RESOURCE_EXHAUSTED.
_MODELS = ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash")

# Scalar params the LLM is allowed to set. We expose the union across all
# formations and filter per-formation later via the function signature.
_ALLOWED_PARAMS = {
    "radius": float, "outer": float, "inner": float, "spacing": float,
    "angle_deg": float, "z": float, "center_x": float, "center_y": float,
    "side": float, "size": float, "axis": str,
}


@dataclass
class Command:
    formation: str
    n: int
    params: dict = field(default_factory=dict)
    reasoning: str = ""
    source: str = "llm"          # "llm" or "fallback"

    def targets(self):
        """Resolve to an (n, 3) array of world targets."""
        return formations.make(self.formation, self.n, **self.params)


def _load_key() -> str | None:
    """Read GEMINI_API_KEY, letting the project .env win over a stale shell var."""
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True), override=True)
    except ImportError:
        pass
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def _coerce_params(formation: str, raw: dict) -> dict:
    """Keep only params the chosen formation accepts; map center_x/y -> center."""
    fn = formations.REGISTRY[formation]
    accepted = set(inspect.signature(fn).parameters) - {"n"}
    out = {}
    cx = cy = None
    for k, v in (raw or {}).items():
        if k not in _ALLOWED_PARAMS or v is None:
            continue
        try:
            v = _ALLOWED_PARAMS[k](v)
        except (TypeError, ValueError):
            continue
        if k == "center_x":
            cx = v
        elif k == "center_y":
            cy = v
        elif k in accepted:
            out[k] = v
    if (cx is not None or cy is not None) and "center" in accepted:
        out["center"] = (cx or 0.0, cy or 0.0)
    return out


def _fallback(instruction: str, n: int) -> Command:
    """Deterministic keyword matcher used when the LLM is unavailable."""
    text = (instruction or "").lower()
    aliases = {
        "heart": "heart", "love": "heart",
        "star": "star", "circle": "circle", "ring": "circle", "round": "circle",
        "line": "line", "row": "line", "vee": "vee", "v-shape": "vee",
        "v shape": "vee", "wedge": "vee", "arrow": "vee", "flock": "vee",
        "square": "square", "box": "square", "grid": "grid", "block": "grid",
        "fibonacci": "fibonacci", "sunflower": "fibonacci", "phyllotaxis": "fibonacci",
        "golden": "fibonacci", "swirl": "fibonacci", "spiral": "fibonacci",
    }
    pick = next((f for kw, f in aliases.items() if kw in text), "circle")
    return Command(pick, n, {}, "keyword fallback (no LLM)", source="fallback")


_SYSTEM = """You are the commander of a drone swarm. Given a human instruction
and a drone count, choose ONE formation from this list and reasonable parameters:

  circle  — params: radius, center_x, center_y, z
  star    — params: outer, inner, center_x, center_y, z   (outer>inner)
  line    — params: spacing, axis ("x" or "y"), center_x, center_y, z
  vee     — params: spacing, angle_deg, center_x, center_y, z   (a V / flock wedge)
  grid    — params: spacing, center_x, center_y, z
  square  — params: side, center_x, center_y, z   (drones on a square outline)
  heart   — params: size, center_x, center_y, z   (a heart shape; size ~3-5 m)
  fibonacci — params: scale, center_x, center_y, z
              (a sunflower / phyllotaxis spiral disc; golden-angle spacing)

Pick the formation that best matches the instruction. Only set parameters that
are clearly implied (e.g. "wide" -> larger radius/spacing, "tight" -> smaller).
Omit parameters you have no reason to change. Flying altitude z defaults to 1.5;
the field is ~10 m wide so keep radii under ~4 m."""


def _response_schema():
    from google.genai import types
    props = {
        "formation": types.Schema(type="STRING", enum=list(formations.REGISTRY)),
        "reasoning": types.Schema(type="STRING"),
        "params": types.Schema(
            type="OBJECT",
            properties={
                "radius": types.Schema(type="NUMBER"),
                "outer": types.Schema(type="NUMBER"),
                "inner": types.Schema(type="NUMBER"),
                "spacing": types.Schema(type="NUMBER"),
                "angle_deg": types.Schema(type="NUMBER"),
                "z": types.Schema(type="NUMBER"),
                "center_x": types.Schema(type="NUMBER"),
                "center_y": types.Schema(type="NUMBER"),
                "side": types.Schema(type="NUMBER"),
                "size": types.Schema(type="NUMBER"),
                "axis": types.Schema(type="STRING", enum=["x", "y"]),
            },
        ),
    }
    return types.Schema(type="OBJECT", properties=props,
                        required=["formation", "reasoning"])


class Commander:
    """LLM-backed formation selector with a deterministic fallback."""

    # Sentinel so callers can pass api_key="" to force the offline fallback,
    # distinct from api_key=None which means "auto-load from env/.env".
    _AUTO = object()

    def __init__(self, api_key=_AUTO, model: str | None = None):
        self.api_key = _load_key() if api_key is self._AUTO else (api_key or None)
        self.models = (model,) if model else _MODELS
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def plan(self, instruction: str, n: int) -> Command:
        """Map a natural-language instruction to a validated Command."""
        if not self.available:
            return _fallback(instruction, n)
        try:
            return self._plan_llm(instruction, n)
        except Exception as e:                       # noqa: BLE001 — never crash the sim
            cmd = _fallback(instruction, n)
            cmd.reasoning = f"LLM error ({type(e).__name__}); {cmd.reasoning}"
            return cmd

    def _plan_llm(self, instruction: str, n: int) -> Command:
        import json
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            response_mime_type="application/json",
            response_schema=_response_schema(),
            temperature=0.2,
        )
        prompt = f"Drone count: {n}\nInstruction: {instruction}"

        last_err = None
        for model in self.models:
            for attempt in range(3):
                try:
                    resp = self._client.models.generate_content(
                        model=model, contents=prompt, config=config)
                    data = json.loads(resp.text)
                    formation = data["formation"]
                    if formation not in formations.REGISTRY:
                        raise ValueError(f"unknown formation {formation!r}")
                    params = _coerce_params(formation, data.get("params") or {})
                    return Command(formation, n, params,
                                   data.get("reasoning", ""), source="llm")
                except Exception as e:               # noqa: BLE001
                    last_err = e
                    if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                        time.sleep(1.5 * (attempt + 1))
                        if attempt == 2:
                            break                    # move to next model
                    else:
                        raise
        raise last_err if last_err else RuntimeError("no model produced a plan")


if __name__ == "__main__":
    import sys
    cmd = Commander()
    print("LLM available:", cmd.available)
    instr = " ".join(sys.argv[1:]) or "form a tight defensive circle"
    plan = cmd.plan(instr, 8)
    print(f"instruction : {instr}")
    print(f"formation   : {plan.formation}  ({plan.source})")
    print(f"params      : {plan.params}")
    print(f"reasoning   : {plan.reasoning}")
    print(f"targets[0]  : {plan.targets()[0]}")
