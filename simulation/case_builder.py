"""
Build an OpenFOAM 11 case directory from a gmsh .msh file and a natural-language prompt.

Workflow
--------
1.  convert_mesh()       – gmshToFoam converts .msh → constant/polyMesh/
2.  read_patches()       – parse constant/polyMesh/boundary
3.  detect_2d()          – check z-extent of the mesh
4.  classify_patch()     – assign role (inlet / outlet / wall / …) from name
5.  update_boundary()    – set correct OpenFOAM boundary types (wall, empty, …)
6.  write_0/             – U and p fields with appropriate BCs
7.  write_constant/      – physicalProperties, momentumTransport
8.  write_system/        – controlDict, fvSchemes, fvSolution

Entry point
-----------
    build_case(case_dir, mesh_path, prompt) -> dict with build metadata
"""

import math
import os
import re
import subprocess
import logging

logger = logging.getLogger(__name__)

# ── OpenFOAM environment ──────────────────────────────────────────────────────
OF_BASHRC = "/opt/openfoam11/etc/bashrc"

def of_env():
    """Return environment dict with OpenFOAM paths sourced."""
    cmd = f"bash -c 'source {OF_BASHRC} && env'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # Start from current env, then overlay with OF env so OF paths win
    env = dict(os.environ)
    for line in result.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            env[k] = v
    return env


_OF_ENV_CACHE = None

def get_of_env():
    global _OF_ENV_CACHE
    if _OF_ENV_CACHE is None:
        _OF_ENV_CACHE = of_env()
    return _OF_ENV_CACHE


# ── Parameter extraction from prompt ─────────────────────────────────────────

def _parse_length_m(prompt: str, keywords: list) -> float | None:
    """
    Search for a length value (in m) near any of the given keywords.
    Handles mm / cm / m suffixes. Returns None if not found.
    """
    p = prompt.lower()
    for kw in keywords:
        m = re.search(
            rf'{re.escape(kw)}\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m\b)?',
            p
        )
        if m:
            val = float(m.group(1))
            unit = (m.group(2) or "m").strip()
            if unit == "mm":
                val /= 1000.0
            elif unit == "cm":
                val /= 100.0
            return val
    return None


def _char_length(prompt: str) -> float | None:
    """
    Infer characteristic length from the prompt for Re → U conversion.
    Returns None if nothing useful is found (caller should keep U default).
    """
    p = prompt.lower()

    # Diameter / radius (cylinder, pipe, sphere)
    L = _parse_length_m(prompt, ["diameter", "diam"])
    if L:
        return L
    r = _parse_length_m(prompt, ["radius"])
    if r:
        return r * 2

    # Chord (airfoil)
    L = _parse_length_m(prompt, ["chord"])
    if L:
        return L

    # Cavity / channel: side, width, height (default 0.1 m if unspecified)
    if any(k in p for k in ("cavity", "lid")):
        L = _parse_length_m(prompt, ["side", "width", "height", "size", "length"])
        return L if L else 0.1

    # Step: step height is the characteristic scale
    if any(k in p for k in ("step", "backward")):
        L = _parse_length_m(prompt, ["step_height", "step height", "step"])
        if L:
            return L

    # Generic fallback: first length-like value in the prompt
    m = re.search(r'(\d+(?:\.\d+)?)\s*(mm|cm)\b', p)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        return val / 1000.0 if unit == "mm" else val / 100.0

    return None


def extract_sim_params(prompt: str) -> dict:
    """
    Extract simulation parameters from a natural-language prompt.

    Returns
    -------
    dict with keys:
        U_mag   – inlet speed  [m/s]   (default 1.0, or computed from Re)
        AoA_deg – angle of attack [°]  (default 0.0, airfoil only)
        nu      – kinematic viscosity  (default 1e-5, air at ~15 °C)
        Re      – Reynolds number if given (optional)
        L_char  – characteristic length used for Re→U conversion (optional)
        n_iter  – number of pseudo-time steps (default 500)
    """
    p = prompt.lower()

    # ── explicit velocity ─────────────────────────────────────────────────────
    U_explicit = None
    m = re.search(r'(\d+(?:\.\d+)?)\s*m/?s', p)
    if m:
        U_explicit = float(m.group(1))

    # ── angle of attack ───────────────────────────────────────────────────────
    AoA_deg = 0.0
    m = re.search(r'(?:aoa|angle\s+of\s+attack|alpha)\s*[=:]?\s*(-?\d+(?:\.\d+)?)', p)
    if m:
        AoA_deg = float(m.group(1))

    # ── Reynolds number ────────────────────────────────────────────────────────
    Re = None
    # Match "Re=1000", "Re 1000", "Reynolds 1000" (number after keyword)
    m = re.search(r're(?:ynolds)?\s*[=:≈~]?\s*([\d.]+(?:e[+-]?\d+)?)', p)
    if m:
        Re = float(m.group(1))
    else:
        # Also match "1000 Re", "1000Re", "1,000 Re" (number before keyword)
        m = re.search(r'([\d.]+(?:e[+-]?\d+)?)\s*re(?:ynolds)?\b', p)
        if m:
            Re = float(m.group(1))

    # ── kinematic viscosity ────────────────────────────────────────────────────
    nu = 1e-5  # air at ~15 °C
    m = re.search(r'nu\s*[=:]\s*([\d.]+(?:e[+-]?\d+)?)', p)
    if m:
        nu = float(m.group(1))
    elif 'water' in p:
        nu = 1e-6
    elif 'oil' in p:
        nu = 1e-4

    # ── U from Re when not explicitly given ───────────────────────────────────
    L_char = None
    if U_explicit is not None:
        U_mag = U_explicit
        # Back-compute L if Re also given, so callers can reference it
        if Re is not None:
            L_char = Re * nu / U_mag
    elif Re is not None:
        L_char = _char_length(prompt)
        if L_char is not None:
            U_mag = Re * nu / L_char
        else:
            # No length found — keep Re constraint by fixing L=1 m convention
            U_mag = Re * nu   # equivalent to L=1 m
            L_char = 1.0
    else:
        U_mag = 1.0

    # ── iterations ────────────────────────────────────────────────────────────
    n_iter = 500

    # ── turbulence model ──────────────────────────────────────────────────────
    turb_model = "laminar"
    if "spalart" in p or "spalartallmaras" in p or "nutilda" in p:
        turb_model = "SpalartAllmaras"
    elif "kepsilon" in p or "k-epsilon" in p or "k epsilon" in p:
        turb_model = "kEpsilon"
    elif "komegasst" in p or "k-omega-sst" in p or "k-omega sst" in p or "komega sst" in p:
        turb_model = "kOmegaSST"
    elif "komega" in p or "k-omega" in p or "k omega" in p:
        turb_model = "kOmegaSST"
    elif "rans" in p or "turbulent" in p:
        turb_model = "kOmegaSST"   # sensible default for RANS

    return {"U_mag": U_mag, "AoA_deg": AoA_deg, "nu": nu, "Re": Re,
            "L_char": L_char, "n_iter": n_iter, "turb_model": turb_model}


# ── Patch reading ─────────────────────────────────────────────────────────────

def read_patches(case_dir: str) -> list[dict]:
    """
    Parse constant/polyMesh/boundary and return a list of patch dicts:
        {name, type, nFaces, startFace}
    Patches with nFaces == 0 are dropped.
    """
    boundary_file = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if not os.path.exists(boundary_file):
        raise FileNotFoundError(f"boundary file not found: {boundary_file}")

    text = open(boundary_file).read()
    # Remove C++ style comments
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    patches = []
    # Match patch blocks: name { ... }
    for m in re.finditer(r'(\w+)\s*\{([^}]*)\}', text):
        name = m.group(1)
        if name in ('FoamFile', 'boundary'):
            continue
        body = m.group(2)
        def _val(key):
            vm = re.search(rf'{key}\s+(\S+)\s*;', body)
            return vm.group(1) if vm else ''
        n_faces = int(_val('nFaces') or '0')
        if n_faces == 0:
            continue
        patches.append({
            'name': name,
            'type': _val('type'),
            'nFaces': n_faces,
            'startFace': int(_val('startFace') or '0'),
        })
    return patches


# ── 2D detection ──────────────────────────────────────────────────────────────

def detect_2d(case_dir: str) -> bool:
    """
    Return True if the mesh is essentially 2D (z-extent < 1 % of x/y span,
    OR a 'defaultFaces' patch with large nFaces exists).
    """
    points_file = os.path.join(case_dir, "constant", "polyMesh", "points")
    if not os.path.exists(points_file):
        return False
    text = open(points_file).read()
    coords = re.findall(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)', text)
    if not coords:
        return False
    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    zs = [float(c[2]) for c in coords]
    span_xy = max(max(xs) - min(xs), max(ys) - min(ys), 1e-12)
    span_z  = max(zs) - min(zs)
    return span_z < 0.05 * span_xy


# ── Patch classification ───────────────────────────────────────────────────────

def classify_patch(name: str, is_2d: bool) -> str:
    """
    Return a role string for BC generation:
        velocity_inlet | pressure_outlet | wall | moving_wall |
        freestream | empty | symmetry | internal
    """
    n = name.lower()

    # Internal / volume patches (gmshToFoam sometimes creates these)
    if n in ('fluid', 'volume', 'interior', 'internal'):
        return 'internal'

    # 2D front/back – empty only for truly 2D (single-cell-in-z) meshes
    if n in ('defaultfaces', 'frontandback'):
        return 'empty'
    if is_2d and n in ('front', 'back'):
        return 'empty'
    if not is_2d and n in ('front', 'back'):
        return 'symmetry'

    # Freestream (far-field) – airfoil / external flow
    if re.search(r'far\s*field|farfield|freestream|far_field', n):
        return 'freestream'

    # Inlet  ──────────────────────────────────────────────────────────────────
    if re.search(r'^inlet', n):
        return 'velocity_inlet'

    # Outlet ──────────────────────────────────────────────────────────────────
    if re.search(r'^outlet', n):
        return 'pressure_outlet'

    # Lid (moving wall – cavity)
    if n == 'lid':
        return 'moving_wall'

    # Solid walls ─────────────────────────────────────────────────────────────
    if re.search(r'wall|cylinder|airfoil|sphere|blade|body', n):
        return 'wall'

    # Symmetry / slip ─────────────────────────────────────────────────────────
    if re.search(r'symm|sym\b|top|bottom|sides?$', n):
        return 'symmetry'

    # Default for unknown: treat as wall (safe)
    return 'wall'


# ── Boundary file update ───────────────────────────────────────────────────────

_FOAM_BOUNDARY_TYPE = {
    'velocity_inlet':  'patch',
    'pressure_outlet': 'patch',
    'wall':            'wall',
    'moving_wall':     'wall',
    'freestream':      'patch',
    'empty':           'empty',
    'symmetry':        'symmetry',
    'internal':        'internal',
}


def update_boundary_types(case_dir: str, patches: list[dict], is_2d: bool):
    """Rewrite constant/polyMesh/boundary with correct OF boundary types."""
    boundary_file = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    text = open(boundary_file).read()

    for p in patches:
        role = classify_patch(p['name'], is_2d)
        foam_type = _FOAM_BOUNDARY_TYPE.get(role, 'patch')
        # Replace `type  <anything>;` inside this patch block
        # We do a targeted replacement for each patch name
        pattern = rf'({re.escape(p["name"])}\s*\{{[^}}]*?)type\s+\w+\s*;'
        replacement = rf'\1type            {foam_type};'
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    open(boundary_file, 'w').write(text)


# ── OpenFOAM file header ───────────────────────────────────────────────────────

def _foam_header(foam_class: str, location: str, object_: str) -> str:
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  11\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n{\n"
        f"    format      ascii;\n"
        f"    class       {foam_class};\n"
        f"    location    \"{location}\";\n"
        f"    object      {object_};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
    )


# ── 0/U ───────────────────────────────────────────────────────────────────────

def write_U(case_dir: str, patches: list[dict], is_2d: bool,
            U_mag: float, AoA_deg: float):
    """Write 0/U with BCs derived from patch roles."""
    AoA_rad = math.radians(AoA_deg)
    Ux = U_mag * math.cos(AoA_rad)
    Uy = U_mag * math.sin(AoA_rad)
    U_str = f"({Ux:.4g} {Uy:.4g} 0)"

    lines = [_foam_header("volVectorField", "0", "U")]
    lines.append("dimensions      [0 1 -1 0 0 0 0];\n")
    lines.append(f"internalField   uniform {U_str};\n")
    lines.append("\nboundaryField\n{\n")

    for p in patches:
        role = classify_patch(p['name'], is_2d)
        if role == 'internal':
            continue
        lines.append(f"    {p['name']}\n    {{\n")
        if role == 'velocity_inlet':
            lines.append(f"        type            fixedValue;\n")
            lines.append(f"        value           uniform {U_str};\n")
        elif role == 'pressure_outlet':
            lines.append(f"        type            inletOutlet;\n")
            lines.append(f"        inletValue      uniform (0 0 0);\n")
            lines.append(f"        value           uniform (0 0 0);\n")
        elif role == 'wall':
            lines.append(f"        type            noSlip;\n")
        elif role == 'moving_wall':
            lines.append(f"        type            fixedValue;\n")
            lines.append(f"        value           uniform ({U_mag:.4g} 0 0);\n")
        elif role == 'freestream':
            lines.append(f"        type            freestreamVelocity;\n")
            lines.append(f"        freestreamValue uniform {U_str};\n")
            lines.append(f"        value           uniform {U_str};\n")
        elif role == 'empty':
            lines.append(f"        type            empty;\n")
        elif role == 'symmetry':
            lines.append(f"        type            symmetry;\n")
        else:
            lines.append(f"        type            zeroGradient;\n")
        lines.append("    }\n\n")

    lines.append("}\n\n")
    lines.append("// " + "*" * 73 + " //\n")

    os.makedirs(os.path.join(case_dir, "0"), exist_ok=True)
    with open(os.path.join(case_dir, "0", "U"), "w") as f:
        f.write("".join(lines))


# ── 0/p ───────────────────────────────────────────────────────────────────────

def write_p(case_dir: str, patches: list[dict], is_2d: bool):
    """Write 0/p with BCs derived from patch roles."""
    lines = [_foam_header("volScalarField", "0", "p")]
    lines.append("dimensions      [0 2 -2 0 0 0 0];\n")
    lines.append("internalField   uniform 0;\n")
    lines.append("\nboundaryField\n{\n")

    has_outlet = any(classify_patch(p['name'], is_2d) == 'pressure_outlet'
                     for p in patches if classify_patch(p['name'], is_2d) != 'internal')

    for p in patches:
        role = classify_patch(p['name'], is_2d)
        if role == 'internal':
            continue
        lines.append(f"    {p['name']}\n    {{\n")
        if role == 'velocity_inlet':
            lines.append(f"        type            zeroGradient;\n")
        elif role == 'pressure_outlet':
            lines.append(f"        type            fixedValue;\n")
            lines.append(f"        value           uniform 0;\n")
        elif role in ('wall', 'moving_wall'):
            lines.append(f"        type            zeroGradient;\n")
        elif role == 'freestream':
            lines.append(f"        type            freestreamPressure;\n")
            lines.append(f"        freestreamValue uniform 0;\n")
        elif role == 'empty':
            lines.append(f"        type            empty;\n")
        elif role == 'symmetry':
            lines.append(f"        type            symmetry;\n")
        else:
            lines.append(f"        type            zeroGradient;\n")
        lines.append("    }\n\n")

    lines.append("}\n\n")
    lines.append("// " + "*" * 73 + " //\n")

    with open(os.path.join(case_dir, "0", "p"), "w") as f:
        f.write("".join(lines))


# ── constant/physicalProperties ───────────────────────────────────────────────

def write_physical_properties(case_dir: str, nu: float):
    os.makedirs(os.path.join(case_dir, "constant"), exist_ok=True)
    content = (
        _foam_header("dictionary", "constant", "physicalProperties") +
        "viscosityModel  constant;\n\n"
        f"nu              [0 2 -1 0 0 0 0] {nu:.6g};\n\n"
        "// " + "*" * 73 + " //\n"
    )
    with open(os.path.join(case_dir, "constant", "physicalProperties"), "w") as f:
        f.write(content)


# ── constant/momentumTransport ────────────────────────────────────────────────

def write_momentum_transport(case_dir: str, model: str = "laminar"):
    """Write momentumTransport for laminar or RANS turbulence models."""
    hdr = _foam_header("dictionary", "constant", "momentumTransport")
    tail = "\n// " + "*" * 73 + " //\n"
    if model == "laminar":
        content = hdr + "simulationType  laminar;\n" + tail
    else:
        content = (
            hdr +
            "simulationType  RAS;\n\n"
            "RAS\n{\n"
            f"    model           {model};\n"
            "    turbulence      on;\n"
            "    printCoeffs     on;\n"
            "}\n" + tail
        )
    with open(os.path.join(case_dir, "constant", "momentumTransport"), "w") as f:
        f.write(content)


# ── 0/k, 0/epsilon, 0/omega, 0/nut (turbulence initial conditions) ────────────

def write_turbulence_fields(case_dir: str, patches: list[dict], is_2d: bool,
                             U_mag: float = 1.0, turb_intensity: float = 0.05,
                             length_scale: float = 0.01):
    """Write k, epsilon, omega, nut fields for RANS simulations."""
    os.makedirs(os.path.join(case_dir, "0"), exist_ok=True)

    k_val     = 1.5 * (U_mag * turb_intensity) ** 2
    eps_val   = 0.09 * k_val ** 1.5 / length_scale
    omega_val = k_val ** 0.5 / (0.09 ** 0.25 * length_scale)
    nut_val   = k_val / omega_val

    def _bc(patches, wall_type, wall_extra, inlet_type, inlet_val, internal_field_val):
        lines = "boundaryField\n{\n"
        for p in patches:
            name = p["name"] if isinstance(p, dict) else p
            role = p.get("role", classify_patch(name, is_2d)) if isinstance(p, dict) else classify_patch(name, is_2d)
            if role == "velocity_inlet":
                lines += f"    {name}\n    {{\n        type    {inlet_type};\n{inlet_val}    }}\n"
            elif role == "freestream":
                # Mixed inlet/outlet — use inletOutlet so both directions work
                lines += (f"    {name}\n    {{\n        type        inletOutlet;\n"
                          f"        inletValue  uniform {internal_field_val};\n"
                          f"        value       $internalField;\n    }}\n")
            elif role in ("wall", "moving_wall"):
                lines += f"    {name}\n    {{\n        type    {wall_type};\n{wall_extra}        value   $internalField;\n    }}\n"
            elif role in ("pressure_outlet",):
                lines += (f"    {name}\n    {{\n        type        inletOutlet;\n"
                          f"        inletValue  uniform {internal_field_val};\n"
                          f"        value       $internalField;\n    }}\n")
            elif role in ("empty", "symmetry", "internal"):
                lines += f"    {name}\n    {{\n        type    {role if role != 'internal' else 'empty'};\n    }}\n"
            else:
                lines += f"    {name}\n    {{\n        type    zeroGradient;\n    }}\n"
        lines += "}\n"
        return lines

    # 0/k
    with open(os.path.join(case_dir, "0", "k"), "w") as f:
        f.write(_foam_header("volScalarField", "0", "k") +
                f"dimensions      [0 2 -2 0 0 0 0];\ninternalField   uniform {k_val:.6g};\n" +
                _bc(patches,
                    "kqRWallFunction", "",
                    "turbulentIntensityKineticEnergyInlet",
                    f"        intensity   {turb_intensity};\n        value       uniform {k_val:.6g};\n",
                    f"{k_val:.6g}") +
                "\n// " + "*" * 73 + " //\n")

    # 0/epsilon
    with open(os.path.join(case_dir, "0", "epsilon"), "w") as f:
        f.write(_foam_header("volScalarField", "0", "epsilon") +
                f"dimensions      [0 2 -3 0 0 0 0];\ninternalField   uniform {eps_val:.6g};\n" +
                _bc(patches,
                    "epsilonWallFunction", "",
                    "turbulentMixingLengthDissipationRateInlet",
                    f"        mixingLength    {length_scale:.6g};\n        value           uniform {eps_val:.6g};\n",
                    f"{eps_val:.6g}") +
                "\n// " + "*" * 73 + " //\n")

    # 0/omega
    with open(os.path.join(case_dir, "0", "omega"), "w") as f:
        f.write(_foam_header("volScalarField", "0", "omega") +
                f"dimensions      [0 0 -1 0 0 0 0];\ninternalField   uniform {omega_val:.6g};\n" +
                _bc(patches,
                    "omegaWallFunction", "",
                    "turbulentMixingLengthFrequencyInlet",
                    f"        mixingLength    {length_scale:.6g};\n        value           uniform {omega_val:.6g};\n",
                    f"{omega_val:.6g}") +
                "\n// " + "*" * 73 + " //\n")

    # 0/nut (computed field, just needs a file with wall functions)
    nut_bc = "boundaryField\n{\n"
    for p in patches:
        name = p["name"] if isinstance(p, dict) else p
        role = p.get("role", classify_patch(name, is_2d)) if isinstance(p, dict) else classify_patch(name, is_2d)
        if role in ("wall", "moving_wall"):
            nut_bc += f"    {name}\n    {{\n        type    nutkWallFunction;\n        value   uniform 0;\n    }}\n"
        elif role in ("empty", "symmetry"):
            nut_bc += f"    {name}\n    {{\n        type    {role};\n    }}\n"
        else:
            nut_bc += f"    {name}\n    {{\n        type    calculated;\n        value   uniform {nut_val:.6g};\n    }}\n"
    nut_bc += "}\n"
    with open(os.path.join(case_dir, "0", "nut"), "w") as f:
        f.write(_foam_header("volScalarField", "0", "nut") +
                f"dimensions      [0 2 -1 0 0 0 0];\ninternalField   uniform {nut_val:.6g};\n" +
                nut_bc + "\n// " + "*" * 73 + " //\n")


def _detect_turb_model(case_dir: str) -> str:
    """Read constant/momentumTransport and return the turbulence model name."""
    mt_path = os.path.join(case_dir, "constant", "momentumTransport")
    if not os.path.exists(mt_path):
        return "laminar"
    with open(mt_path) as f:
        text = f.read()
    text_l = text.lower()
    if "kepsilon" in text_l or "k-epsilon" in text_l:
        return "kEpsilon"
    if "komegasst" in text_l or "k-omega-sst" in text_l:
        return "kOmegaSST"
    if "komega" in text_l or "k-omega" in text_l:
        return "kOmega"
    if "spalart" in text_l or "nutilda" in text_l:
        return "SpalartAllmaras"
    return "laminar"


# ── system/controlDict ────────────────────────────────────────────────────────

def write_control_dict(case_dir: str, n_iter: int):
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
    content = (
        _foam_header("dictionary", "system", "controlDict") +
        "application     foamRun;\n\n"
        "solver          incompressibleFluid;\n\n"
        "startFrom       startTime;\n\n"
        "startTime       0;\n\n"
        "stopAt          endTime;\n\n"
        f"endTime         {n_iter};\n\n"
        "deltaT          1;\n\n"
        "writeControl    timeStep;\n\n"
        f"writeInterval   {max(50, n_iter // 4)};\n\n"
        "purgeWrite      2;\n\n"
        "writeFormat     ascii;\n\n"
        "writePrecision  6;\n\n"
        "runTimeModifiable yes;\n\n"
        "// " + "*" * 73 + " //\n"
    )
    with open(os.path.join(case_dir, "system", "controlDict"), "w") as f:
        f.write(content)


# ── system/fvSchemes ──────────────────────────────────────────────────────────

def write_fv_schemes(case_dir: str):
    content = (
        _foam_header("dictionary", "system", "fvSchemes") +
        "ddtSchemes\n{\n    default         Euler;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
        "divSchemes\n{\n"
        "    default         none;\n"
        "    div(phi,U)      Gauss linearUpwind grad(U);\n"
        "}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n\n"
        "// " + "*" * 73 + " //\n"
    )
    with open(os.path.join(case_dir, "system", "fvSchemes"), "w") as f:
        f.write(content)


# ── system/fvSolution ─────────────────────────────────────────────────────────

def write_fv_solution(case_dir: str):
    content = (
        _foam_header("dictionary", "system", "fvSolution") +
        "solvers\n{\n"
        "    p\n    {\n"
        "        solver          GAMG;\n"
        "        tolerance       1e-06;\n"
        "        relTol          0.1;\n"
        "        smoother        GaussSeidel;\n"
        "    }\n\n"
        "    pFinal\n    {\n"
        "        $p;\n"
        "        tolerance       1e-06;\n"
        "        relTol          0;\n"
        "    }\n\n"
        "    \"(U|k|epsilon|omega|nuTilda).*\"\n    {\n"
        "        solver          smoothSolver;\n"
        "        smoother        GaussSeidel;\n"
        "        tolerance       1e-05;\n"
        "        relTol          0;\n"
        "    }\n"
        "}\n\n"
        "PIMPLE\n{\n"
        "    nCorrectors              2;\n"
        "    nNonOrthogonalCorrectors 1;\n"
        "    pRefCell                 0;\n"
        "    pRefValue                0;\n"
        "}\n\n"
        "relaxationFactors\n{\n"
        "    fields  { p 0.3; }\n"
        "    equations { U 0.7; }\n"
        "}\n\n"
        "// " + "*" * 73 + " //\n"
    )
    with open(os.path.join(case_dir, "system", "fvSolution"), "w") as f:
        f.write(content)


def write_fv_solution_rans(case_dir: str):
    """Write fvSolution configured for steady RANS (SIMPLE algorithm)."""
    content = (
        _foam_header("dictionary", "system", "fvSolution") +
        "solvers\n{\n"
        "    p\n    {\n"
        "        solver          GAMG;\n"
        "        tolerance       1e-07;\n"
        "        relTol          0.05;\n"
        "        smoother        GaussSeidel;\n"
        "    }\n\n"
        "    \"(U|k|epsilon|omega|nuTilda).*\"\n    {\n"
        "        solver          smoothSolver;\n"
        "        smoother        GaussSeidel;\n"
        "        tolerance       1e-07;\n"
        "        relTol          0.1;\n"
        "    }\n"
        "}\n\n"
        "SIMPLE\n{\n"
        "    nNonOrthogonalCorrectors 2;\n"
        "    consistent      yes;\n"
        "    pRefCell        0;\n"
        "    pRefValue       0;\n"
        "}\n\n"
        "relaxationFactors\n{\n"
        "    fields  { p 0.3; }\n"
        "    equations\n    {\n"
        "        U       0.7;\n"
        "        k       0.5;\n"
        "        epsilon 0.5;\n"
        "        omega   0.5;\n"
        "        nuTilda 0.5;\n"
        "    }\n"
        "}\n\n"
        "// " + "*" * 73 + " //\n"
    )
    with open(os.path.join(case_dir, "system", "fvSolution"), "w") as f:
        f.write(content)


def write_fv_schemes_rans(case_dir: str):
    """Write fvSchemes for steady RANS (steady-state ddtSchemes)."""
    content = (
        _foam_header("dictionary", "system", "fvSchemes") +
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n}\n\n"
        "divSchemes\n{\n"
        "    default         none;\n"
        "    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
        "    div(phi,k)      bounded Gauss upwind;\n"
        "    div(phi,epsilon) bounded Gauss upwind;\n"
        "    div(phi,omega)  bounded Gauss upwind;\n"
        "    div(phi,nuTilda) bounded Gauss upwind;\n"
        "    div((nuEff*dev(T(grad(U))))) Gauss linear;\n"
        "}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         corrected;\n}\n\n"
        "wallDist\n{\n    method  meshWave;\n}\n\n"
        "// " + "*" * 73 + " //\n"
    )
    with open(os.path.join(case_dir, "system", "fvSchemes"), "w") as f:
        f.write(content)


# ── Master entry point ────────────────────────────────────────────────────────

def build_case(case_dir: str, mesh_path: str, prompt: str) -> dict:
    """
    Build a complete OpenFOAM case directory.

    Parameters
    ----------
    case_dir   : target directory (will be created if needed)
    mesh_path  : path to the .msh file from Phase 1
    prompt     : original natural-language prompt

    Returns
    -------
    dict with keys: case_dir, patches, is_2d, sim_params, error (or None)
    """
    os.makedirs(case_dir, exist_ok=True)

    # 1. Extract simulation parameters from prompt
    sim_params = extract_sim_params(prompt)
    logger.info(f"Sim params: {sim_params}")

    # 2. Write minimal system/controlDict before gmshToFoam (OF requires it)
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
    write_control_dict(case_dir, sim_params["n_iter"])

    # 3. Convert mesh
    env = get_of_env()
    gmsh_cmd = ["gmshToFoam", "-case", case_dir, mesh_path]
    logger.info(f"Running: {' '.join(gmsh_cmd)}")
    result = subprocess.run(gmsh_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        err = (result.stderr or result.stdout)[-800:]
        return {"case_dir": case_dir, "error": f"gmshToFoam failed: {err}"}
    logger.info("gmshToFoam OK")

    # 3. Detect 2D / 3D
    is_2d = detect_2d(case_dir)
    logger.info(f"2D mesh detected: {is_2d}")

    # 4. Read patches
    try:
        patches = read_patches(case_dir)
    except Exception as e:
        return {"case_dir": case_dir, "error": f"read_patches failed: {e}"}
    logger.info(f"Patches: {[p['name'] for p in patches]}")

    # 5. Update boundary types
    update_boundary_types(case_dir, patches, is_2d)

    # 6. Write field files
    write_U(case_dir, patches, is_2d, sim_params["U_mag"], sim_params["AoA_deg"])
    write_p(case_dir, patches, is_2d)

    # 7. Write constant/
    write_physical_properties(case_dir, sim_params["nu"])
    write_momentum_transport(case_dir)

    # 8. Write system/
    write_control_dict(case_dir, sim_params["n_iter"])
    write_fv_schemes(case_dir)
    write_fv_solution(case_dir)

    return {
        "case_dir": case_dir,
        "patches": [p["name"] for p in patches],
        "is_2d": is_2d,
        "sim_params": sim_params,
        "error": None,
    }
