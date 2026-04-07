"""
Template-based OpenFOAM file patching.

Instead of generating full OF files from scratch (fragile), we:
1. Retrieve the most similar tutorial file as a base template (guaranteed correct syntax)
2. LLM outputs a compact JSON patch spec (only what needs to change)
3. This module applies the patch to the template

JSON patch spec for 0/U:
{
  "internalField": "uniform (10 0 0)",
  "boundaryField": {
    "inlet":   {"type": "fixedValue",  "value": "uniform (10 0 0)"},
    "outlet":  {"type": "inletOutlet", "inletValue": "uniform (0 0 0)"},
    "walls":   {"type": "noSlip"},
    "front":   {"type": "empty"},
    "back":    {"type": "empty"}
  }
}

For scalar fields (0/p, 0/k, etc.) internalField is a scalar, e.g. "uniform 0".
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def _find_block_end(text: str, start: int) -> int:
    """Return index of closing } for block starting at `start` (first {)."""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text) - 1


def _replace_internal_field(text: str, value: str) -> str:
    """Replace the internalField line."""
    return re.sub(
        r'(internalField\s+).*?;',
        lambda m: f"{m.group(1)}{value};",
        text,
        count=1,
    )


def _replace_boundary_field(text: str, patch_specs: dict) -> str:
    """
    Replace boundaryField block in text with one built from patch_specs.
    patch_specs: {patch_name: {type: ..., value: ..., ...}}
    """
    # Find boundaryField block
    m = re.search(r'\bboundaryField\s*\{', text)
    if not m:
        return text

    block_start = m.start()
    brace_start = text.index('{', m.start())
    block_end   = _find_block_end(text, brace_start)

    # Build new boundaryField block
    lines = ["boundaryField", "{"]
    for patch_name, bc in patch_specs.items():
        lines.append(f"    {patch_name}")
        lines.append("    {")
        for key, val in bc.items():
            lines.append(f"        {key:<16}{val};")
        lines.append("    }")
        lines.append("")
    lines.append("}")

    new_block = "\n".join(lines)
    return text[:block_start] + new_block + text[block_end + 1:]


def apply_patch(template: str, patch: dict) -> str:
    """
    Apply a JSON patch spec to a template OF file.

    patch keys:
      "internalField"  → str, e.g. "uniform (0 0 0)"
      "internalValue"  → alias for internalField
      "boundaryField"  → dict of {patch_name: {type, value, ...}}
      "dimensions"     → str, e.g. "[0 1 -1 0 0 0 0]"
    """
    result = template

    # dimensions
    if "dimensions" in patch:
        result = re.sub(
            r'dimensions\s+\[[^\]]*\]\s*;',
            f"dimensions      {patch['dimensions']};",
            result,
            count=1,
        )

    # internalField
    internal = patch.get("internalField") or patch.get("internalValue")
    if internal:
        result = _replace_internal_field(result, internal)

    # boundaryField
    if "boundaryField" in patch:
        result = _replace_boundary_field(result, patch["boundaryField"])

    return result


# ── JSON patch extraction ─────────────────────────────────────────────────────

def extract_patch_json(llm_output: str) -> dict | None:
    """
    Extract JSON patch spec from LLM output.
    Handles both raw JSON and JSON wrapped in markdown code fences.
    Returns dict or None if parsing fails.
    """
    text = llm_output.strip()

    # Try to extract from ```json ... ``` fence
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        text = m.group(1)

    # Find first '{"' — a real JSON object start (skip description text like {type, ...})
    # Try '{"' first, fall back to any '{'
    start = text.find('{"')
    if start == -1:
        start = text.find('{')
    if start == -1:
        return None
    end = _find_block_end(text, start)
    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse patch JSON: {candidate[:200]}")
        return None


# ── patch spec assembler ──────────────────────────────────────────────────────

def build_patch_prompt(slot: str, template: str, prompt: str,
                        sim_params: dict, patches: list, is_2d: bool) -> tuple[str, str]:
    """
    Build system + user messages for LLM to produce a JSON patch spec.
    The template is shown; LLM only needs to specify what changes.
    """
    patch_names = [p['name'] if isinstance(p, dict) else p for p in patches]
    dim_hint = ""
    if slot == "0/U":
        u = sim_params.get("U_mag", 1.0)
        aoa = sim_params.get("AoA_deg", 0.0)
        import math
        ux = u * math.cos(math.radians(aoa))
        uy = u * math.sin(math.radians(aoa))
        dim_hint = f"\nVelocity vector for AoA={aoa}°: ({ux:.4f} {uy:.4f} 0)"

    system_msg = (
        f'You are an OpenFOAM 11 expert. Output ONLY a valid JSON object — no text, no explanation.\n\n'
        f'Format:\n'
        f'{{\n'
        f'  "internalField": "uniform (Ux Uy 0)",\n'
        f'  "boundaryField": {{\n'
        f'    "patchName": {{"type": "fixedValue", "value": "uniform (1 0 0)"}},\n'
        f'    "patchName2": {{"type": "noSlip"}}\n'
        f'  }}\n'
        f'}}\n\n'
        f"Valid BC types: fixedValue, noSlip, zeroGradient, inletOutlet, "
        f"freestreamVelocity, freestreamPressure, empty, symmetry, slip.\n"
        f"For inletOutlet include inletValue. "
        f"Use ONLY the patch names listed in the prompt."
    )

    user_msg = (
        f"Simulation: {prompt}\n"
        f"Parameters: U_mag={sim_params.get('U_mag', 1.0)} m/s, "
        f"nu={sim_params.get('nu', 1e-5)}, Re={sim_params.get('Re', 1000)}, "
        f"AoA={sim_params.get('AoA_deg', 0.0)} deg, "
        f"{'2D' if is_2d else '3D'}{dim_hint}\n\n"
        f"Template ({slot}) — for syntax reference only, its patch names may differ:\n"
        f"{template}\n\n"
        f"THIS CASE has these patch names (use ONLY these, not the template's):\n"
        f"  {', '.join(patch_names)}\n\n"
        f"Output JSON patch spec using patch names: {', '.join(patch_names)}"
    )
    return system_msg, user_msg
