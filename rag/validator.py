"""
Lightweight syntactic validation for generated OpenFOAM files.
"""
import re


# ── BC repair ─────────────────────────────────────────────────────────────────

# Safe default patch blocks per slot and role
_U_DEFAULTS = {
    "velocity_inlet": "type            fixedValue;\n        value           uniform (1 0 0);",
    "pressure_outlet": "type            inletOutlet;\n        inletValue      uniform (0 0 0);\n        value           uniform (0 0 0);",
    "moving_wall":    "type            fixedValue;\n        value           uniform (1 0 0);",
    "wall":           "type            noSlip;",
    "empty":          "type            empty;",
    "symmetry":       "type            symmetry;",
    "default":        "type            noSlip;",
}
_P_DEFAULTS = {
    "velocity_inlet": "type            zeroGradient;",
    "pressure_outlet": "type            fixedValue;\n        value           uniform 0;",
    "moving_wall":    "type            zeroGradient;",
    "wall":           "type            zeroGradient;",
    "empty":          "type            empty;",
    "symmetry":       "type            symmetry;",
    "default":        "type            zeroGradient;",
}


def _extract_named_patches(text: str) -> set:
    """Return set of patch names that have an explicit named block in boundaryField."""
    # Find the boundaryField block
    m = re.search(r'boundaryField\s*\{', text)
    if not m:
        return set()
    start = m.end()
    # Walk to find the matching closing brace
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    bf_content = text[start:i - 1]
    # Find all patch names: word followed by optional whitespace then {
    return set(re.findall(r'\b([A-Za-z_]\w*)\s*\{', bf_content))


def _default_block(patch_name: str, slot: str, classify_fn) -> str:
    """Return a safe default patch block for the given slot and patch role."""
    role = classify_fn(patch_name, False)
    if slot == "0/U":
        body = _U_DEFAULTS.get(role, _U_DEFAULTS["default"])
    else:
        body = _P_DEFAULTS.get(role, _P_DEFAULTS["default"])
    return f"    {patch_name}\n    {{\n        {body}\n    }}\n"


def inject_missing_patches(text: str, slot: str, patches: list, classify_fn) -> str:
    """
    Ensure every mesh patch has a named block in boundaryField.
    For patches that are completely missing (or only appear in unnamed blocks),
    inject a safe default block before the closing brace of boundaryField.
    """
    present = _extract_named_patches(text)
    patch_names = [p["name"] if isinstance(p, dict) else p for p in patches]
    missing = [p for p in patch_names if p not in present]
    if not missing:
        return text

    # Build injection text
    injection = "\n" + "".join(_default_block(p, slot, classify_fn) for p in missing)

    # Insert before the final closing brace of boundaryField
    # Find the boundaryField closing brace position
    m = re.search(r'boundaryField\s*\{', text)
    if not m:
        return text
    depth = 1
    i = m.end()
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    # i is now one past the closing brace
    insert_at = i - 1
    return text[:insert_at] + injection + text[insert_at:]


def remove_unnamed_bc_blocks(text: str) -> str:
    """
    Remove unnamed (anonymous) blocks from boundaryField.
    These are blocks of the form '{ ... }' at the patch level with no name,
    which cause OpenFOAM to crash with 'Cannot find patchField entry for X'.
    """
    m = re.search(r'(boundaryField\s*\{)', text)
    if not m:
        return text

    prefix = text[:m.end()]
    rest = text[m.end():]

    # Split rest into lines and rebuild, skipping anonymous top-level blocks
    result = []
    depth = 1  # we're inside boundaryField {
    i = 0
    lines = rest.splitlines(keepends=True)
    skip_depth = None

    for line in lines:
        stripped = line.strip()
        open_count = line.count('{')
        close_count = line.count('}')

        # Detect anonymous block: line that is only '{' at depth 1
        if skip_depth is None and stripped == '{' and depth == 1:
            skip_depth = depth  # start skipping
            depth += open_count - close_count
            continue

        if skip_depth is not None:
            depth += open_count - close_count
            if depth <= skip_depth:
                skip_depth = None  # done skipping
            continue

        depth += open_count - close_count
        result.append(line)

    return prefix + "".join(result)


def _balanced_braces(text: str) -> bool:  # also importable directly
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


SLOT_REQUIRED = {
    "0/U": ["FoamFile", "dimensions", "boundaryField"],
    "0/p": ["FoamFile", "dimensions", "boundaryField"],
    "0/k": ["FoamFile", "dimensions", "boundaryField"],
    "0/epsilon": ["FoamFile", "dimensions", "boundaryField"],
    "0/omega": ["FoamFile", "dimensions", "boundaryField"],
    "0/nuTilda": ["FoamFile", "dimensions", "boundaryField"],
    "constant/physicalProperties": ["FoamFile", "nu"],
    "constant/momentumTransport":  ["FoamFile", "simulationType"],
    "system/fvSchemes":   ["FoamFile", "ddtSchemes", "divSchemes", "gradSchemes"],
    "system/fvSolution":  ["FoamFile", "solvers"],
    "system/controlDict": ["FoamFile", "application", "endTime", "deltaT"],
}


# Correct dimensions per slot (must appear verbatim after fixup)
SLOT_DIMENSIONS = {
    "0/U":       "[0 1 -1 0 0 0 0]",
    "0/p":       "[0 2 -2 0 0 0 0]",
    "0/k":       "[0 2 -2 0 0 0 0]",
    "0/epsilon": "[0 2 -3 0 0 0 0]",
    "0/omega":   "[0 0 -1 0 0 0 0]",
    "0/nuTilda": "[0 2 -1 0 0 0 0]",
    "0/nut":     "[0 2 -1 0 0 0 0]",
}


def fix_dimensions(text: str, slot: str) -> str:
    """Replace any 'dimensions [...]' line with the correct one for this slot."""
    correct = SLOT_DIMENSIONS.get(slot)
    if not correct:
        return text
    return re.sub(
        r'dimensions\s+\[[^\]]*\]\s*;',
        f"dimensions      {correct};",
        text,
    )


def validate_foam_file(text: str, slot: str) -> tuple:
    """Returns (ok: bool, error_msg: str)."""
    if not text or len(text.strip()) < 30:
        return False, "empty or too short"

    if not _balanced_braces(text):
        return False, "unbalanced braces"

    required = SLOT_REQUIRED.get(slot, ["FoamFile"])
    for token in required:
        if token not in text:
            return False, f"missing required token: {token!r}"

    return True, ""


def validate_patch_coverage(u_text: str, patches: list) -> tuple:
    """Check every patch appears in 0/U boundaryField."""
    missing = [p for p in patches if p not in u_text]
    return (len(missing) == 0), missing


def strip_markdown(text: str) -> str:
    """Remove markdown code fences if the model added them."""
    blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return "\n".join(b.strip() for b in blocks)
    fence = re.match(r"```(?:\w+)?\s*", text.lstrip())
    if fence:
        return text.lstrip()[fence.end():].strip()
    return text.strip()


TURB_SOLVER_BLOCK = """
    "(k|epsilon|omega|nuTilda)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        nSweeps         2;
        tolerance       1e-08;
        relTol          0.1;
    }
"""

def fix_fv_solution_turbulence(text: str, turb_model: str) -> str:
    """
    Inject a wildcard turbulence solver entry into fvSolution if missing.
    Uses a regex pattern that covers k, epsilon, omega, nuTilda in one entry.
    """
    if turb_model == "laminar":
        return text
    # Check if any turbulence field solver already present
    if re.search(r'"?\(k\|epsilon', text) or ('"epsilon"' in text and 'solver' in text.split('"epsilon"')[1][:30]):
        return text
    fields_needed = {"kEpsilon": ("epsilon", "k"), "kOmega": ("omega", "k"),
                     "kOmegaSST": ("omega", "k"), "SpalartAllmaras": ("nuTilda",)}
    required = fields_needed.get(turb_model, ())
    needs_inject = any(f'"{f}"' not in text and f not in text for f in required)
    if needs_inject:
        text = re.sub(
            r'(solvers\s*\{)',
            r'\1' + TURB_SOLVER_BLOCK,
            text,
            count=1,
        )
    return text


def fix_fv_schemes_turbulence(text: str, turb_model: str) -> str:
    """
    Inject missing turbulence transport div-schemes into fvSchemes.
    Called after LLM generation when turb_model != laminar.
    """
    if turb_model == "laminar":
        return text

    needed = []
    if turb_model in ("kEpsilon", "kOmegaSST", "kOmega"):
        needed.append(('div(phi,k)',       'bounded Gauss linearUpwind default'))
    if turb_model == "kEpsilon":
        needed.append(('div(phi,epsilon)', 'bounded Gauss linearUpwind default'))
    if turb_model in ("kOmega", "kOmegaSST"):
        needed.append(('div(phi,omega)',   'bounded Gauss linearUpwind default'))
    if turb_model == "SpalartAllmaras":
        needed.append(('div(phi,nuTilda)', 'bounded Gauss linearUpwind default'))

    for key, scheme in needed:
        if key not in text:
            # Insert before the closing brace of divSchemes block
            text = re.sub(
                r'(divSchemes\s*\{[^}]*)',
                lambda m: m.group(0) + f'\n    {key}  {scheme};',
                text,
                count=1,
                flags=re.DOTALL,
            )

    # kOmegaSST requires wallDist method for y+ computation
    if turb_model in ("kOmegaSST", "kOmega") and "wallDist" not in text:
        text = text.rstrip() + "\n\nwallDist\n{\n    method  meshWave;\n}\n"

    return text


SIMPLE_BLOCK = """
SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega|nuTilda)" 1e-4;
    }
}

relaxationFactors
{
    equations
    {
        U               0.7;
        k               0.7;
        epsilon         0.7;
        omega           0.7;
        nuTilda         0.7;
    }
}
"""


def _remove_block(text: str, keyword: str) -> str:
    """Remove a top-level 'keyword { ... }' block from text (handles nested braces)."""
    result = []
    i = 0
    while i < len(text):
        # Check if we're at the start of the target keyword block
        m = re.match(rf'\s*{re.escape(keyword)}\s*\{{', text[i:])
        if m:
            # Skip to the matching closing brace
            depth = 0
            j = i + m.start() + len(m.group()) - 1  # position of opening {
            while j < len(text):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        i = j + 1
                        break
                j += 1
            else:
                i = len(text)
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def fix_fv_solution_regime(text: str, regime: str) -> str:
    """
    If regime=steady but LLM generated PIMPLE/PISO, replace with SIMPLE block.
    """
    if regime != "steady":
        return text
    if "SIMPLE" in text:
        return text   # already has SIMPLE
    if "PIMPLE" not in text and "PISO" not in text:
        return text   # no transient block to replace
    # Remove PIMPLE/PISO blocks (handles arbitrary nesting depth)
    for kw in ("PIMPLE", "PISO"):
        if kw in text:
            text = _remove_block(text, kw)
    # Append SIMPLE block
    if "SIMPLE" not in text:
        # Insert before final comment line or at end
        insert_pos = text.rfind("// " + "*" * 5)
        if insert_pos > 0:
            text = text[:insert_pos] + SIMPLE_BLOCK + "\n" + text[insert_pos:]
        else:
            text = text.rstrip() + "\n" + SIMPLE_BLOCK + "\n"
    return text


def sanitize_control_dict(text: str) -> str:
    """
    Remove #include directives and functions{} blocks from a controlDict.
    LLM sometimes adds these referencing files that don't exist.
    """
    # Remove #include lines
    text = re.sub(r'^\s*#include\s+"[^"]*"\s*$', '', text, flags=re.MULTILINE)

    # Remove functions {} block (including nested braces)
    # Simple approach: remove everything from "functions" keyword to matching closing brace
    cleaned = []
    depth = 0
    in_functions = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_functions and re.match(r'functions\s*$', stripped):
            in_functions = True
            depth = 0
            continue
        if in_functions:
            depth += stripped.count('{') - stripped.count('}')
            if depth < 0:
                in_functions = False
                depth = 0
            continue
        cleaned.append(line)
    return "\n".join(cleaned)
