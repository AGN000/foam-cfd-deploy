"""
Build system + user messages for each OpenFOAM file slot.
"""

SLOT_SYSTEM_PROMPTS = {
    "0/U": (
        "You are an OpenFOAM 11 expert. Generate a valid 0/U boundary condition file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Use the correct FoamFile header with dimensions [0 1 -1 0 0 0 0]. "
        "Every patch listed in the Patches section must appear in boundaryField."
    ),
    "0/p": (
        "You are an OpenFOAM 11 expert. Generate a valid 0/p boundary condition file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Use dimensions [0 2 -2 0 0 0 0] for kinematic pressure. "
        "Every patch must appear in boundaryField."
    ),
    "0/k": (
        "You are an OpenFOAM 11 expert. Generate a valid 0/k turbulent kinetic energy file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Dimensions are [0 2 -2 0 0 0 0]."
    ),
    "0/epsilon": (
        "You are an OpenFOAM 11 expert. Generate a valid 0/epsilon dissipation rate file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Dimensions are [0 2 -3 0 0 0 0]."
    ),
    "0/omega": (
        "You are an OpenFOAM 11 expert. Generate a valid 0/omega specific dissipation rate file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Dimensions are [0 0 -1 0 0 0 0]."
    ),
    "0/nuTilda": (
        "You are an OpenFOAM 11 expert. Generate a valid 0/nuTilda Spalart-Allmaras field file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Dimensions are [0 2 -1 0 0 0 0]."
    ),
    "constant/physicalProperties": (
        "You are an OpenFOAM 11 expert. Generate a valid constant/physicalProperties file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "This file sets the kinematic viscosity nu."
    ),
    "constant/momentumTransport": (
        "You are an OpenFOAM 11 expert. Generate a valid constant/momentumTransport file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "This file selects the turbulence model (laminar, kEpsilon, kOmegaSST, etc.)."
    ),
    "system/fvSchemes": (
        "You are an OpenFOAM 11 expert. Generate a valid system/fvSchemes file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Must contain ddtSchemes, gradSchemes, divSchemes, laplacianSchemes, interpolationSchemes, snGradSchemes."
    ),
    "system/fvSolution": (
        "You are an OpenFOAM 11 expert. Generate a valid system/fvSolution file. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "Must contain solvers{} block and SIMPLE or PIMPLE block with residualControl."
    ),
    "system/controlDict": (
        "You are an OpenFOAM 11 expert. Generate a valid system/controlDict file for OpenFOAM 11. "
        "Output ONLY the raw OpenFOAM file content — no markdown fences, no explanation. "
        "IMPORTANT: Do NOT include any #include directives. Do NOT include a functions{} block. "
        "Use application foamRun; solver incompressibleFluid; "
        "Must contain startTime, endTime, deltaT, writeInterval."
    ),
}

DEFAULT_SLOTS = [
    "0/U",
    "0/p",
    "constant/physicalProperties",
    "constant/momentumTransport",
    "system/fvSchemes",
    "system/fvSolution",
    "system/controlDict",
]


def assemble_prompt(
    slot: str,
    prompt: str,
    context: str,
    sim_params: dict,
    patches: list,
    is_2d: bool,
    prev_generated: dict = None,
) -> tuple:
    """
    Build (system_message, user_message) for a single file slot.
    prev_generated: already-generated slots (e.g. 0/U text when generating 0/p).
    """
    system = SLOT_SYSTEM_PROMPTS.get(slot, SLOT_SYSTEM_PROMPTS["system/controlDict"])

    U_mag  = sim_params.get("U_mag", 1.0)
    AoA    = sim_params.get("AoA_deg", 0.0)
    nu     = sim_params.get("nu", 1e-5)
    Re     = sim_params.get("Re")
    n_iter = sim_params.get("n_iter", 500)
    dim_str = "quasi-2D (1-cell thick, front/back as empty)" if is_2d else "3D"

    user_parts = [
        f"Simulation: {prompt}",
        f"Parameters: U_mag={U_mag} m/s, nu={nu}, Re={Re or 'auto'}, AoA={AoA} deg, {n_iter} iterations, {dim_str}",
        f"Patches: {', '.join(p['name'] if isinstance(p, dict) else p for p in patches) if patches else 'unknown'}",
        "",
    ]

    if context:
        user_parts += [
            "Reference examples from OpenFOAM tutorials (adapt as needed):",
            context,
        ]

    # Chain-of-slots: inject already-generated U when generating p (helps match patch names)
    if slot == "0/p" and prev_generated and "0/U" in prev_generated:
        user_parts += [
            "Already generated 0/U for this case:",
            prev_generated["0/U"],
            "",
        ]

    user_parts.append(f"Generate the {slot} file for this simulation.")
    user_msg = "\n".join(user_parts)
    return system, user_msg
