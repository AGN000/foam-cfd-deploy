"""
Drop-in replacement for simulation/case_builder.build_case().
Phase A: identical mesh conversion + param extraction.
Phase B: RAG-retrieval → LLM generation → validation.
Falls back to hardcoded case_builder functions on validation failure.
"""
import logging
import os
import subprocess
from pathlib import Path

from simulation.case_builder import (
    extract_sim_params, read_patches, detect_2d,
    update_boundary_types, of_env,
    write_U, write_p, write_physical_properties,
    write_momentum_transport, write_control_dict,
    write_fv_schemes, write_fv_solution,
    write_fv_schemes_rans, write_fv_solution_rans,
    write_turbulence_fields, _detect_turb_model,
)
from rag.validator import (
    fix_fv_schemes_turbulence, fix_fv_solution_turbulence, fix_fv_solution_regime,
    inject_missing_patches, remove_unnamed_bc_blocks,
)

logger = logging.getLogger(__name__)

# Base BC slots always attempted via RAG+LLM
_BASE_RAG_SLOTS = ["0/U", "0/p"]

# Turbulence BC slots — added to RAG when RANS model is detected.
# 0/nut and 0/nuTilda are computed/diagnostic fields with mandatory wall functions;
# always written by the hardcoded fallback for correctness.
_TURB_RAG_SLOTS = ["0/k", "0/epsilon", "0/omega"]

# Hardcoded fallback writers per slot
FALLBACK_WRITERS = {
    "0/U":                         lambda cd, p, s, is2d: write_U(cd, p, is2d, s["U_mag"], s["AoA_deg"]),
    "0/p":                         lambda cd, p, s, is2d: write_p(cd, p, is2d),
    "constant/physicalProperties": lambda cd, p, s, is2d: write_physical_properties(cd, s["nu"]),
    "constant/momentumTransport":  lambda cd, p, s, is2d: write_momentum_transport(cd, s.get("turb_model", "laminar")),
    "system/fvSchemes":            lambda cd, p, s, is2d: (write_fv_schemes_rans(cd) if s.get("turb_model", "laminar") != "laminar" else write_fv_schemes(cd)),
    "system/fvSolution":           lambda cd, p, s, is2d: (write_fv_solution_rans(cd) if s.get("turb_model", "laminar") != "laminar" else write_fv_solution(cd)),
    "system/controlDict":          lambda cd, p, s, is2d: write_control_dict(cd, s["n_iter"]),
}


def build_case_rag(
    case_dir: str,
    mesh_path: str,
    prompt: str,
    rag_retriever=None,
    llm_generator=None,
    fallback: bool = True,
) -> dict:
    """
    Build a complete OpenFOAM case directory.

    If rag_retriever and llm_generator are provided, uses RAG+LLM to
    generate boundary condition files. Falls back to hardcoded writers
    on validation failure (when fallback=True).

    Returns dict: case_dir, patches, is_2d, sim_params,
                  rag_used {slot: bool}, error
    """
    use_rag = (rag_retriever is not None and llm_generator is not None)

    # ── Phase A: mesh conversion (identical to old build_case) ───────────────
    sim_params = extract_sim_params(prompt)
    turb_model = sim_params.get("turb_model", "laminar")

    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)
    write_control_dict(case_dir, sim_params["n_iter"])

    env = of_env()
    cmd = ["gmshToFoam", "-case", case_dir, mesh_path]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if result.returncode != 0:
        return {
            "case_dir": case_dir, "patches": [], "is_2d": False,
            "sim_params": sim_params, "rag_used": {}, "error": result.stderr[:500],
        }

    patches = read_patches(case_dir)
    is_2d   = detect_2d(case_dir)
    update_boundary_types(case_dir, patches, is_2d)

    logger.info(f"Mesh converted: patches={patches}, is_2d={is_2d}")

    # Determine which slots to attempt via RAG
    rag_slots = list(_BASE_RAG_SLOTS)
    if turb_model != "laminar":
        rag_slots += _TURB_RAG_SLOTS

    # ── Phase B: RAG + LLM case file generation ───────────────────────────────
    rag_used = {}
    generated = {}

    if use_rag:
        logger.info(f"Generating case files via RAG + LLM (turb={turb_model})...")
        patch_names = [p["name"] if isinstance(p, dict) else p for p in patches]
        generated = llm_generator.generate_case_files(
            prompt, sim_params, patch_names, is_2d, rag_retriever, slots=rag_slots
        )
        for slot, result_data in generated.items():
            slot_ok = result_data.get("valid", False)
            rag_used[slot] = slot_ok
            # Write successful RAG results now for base slots;
            # turb slots are written after Phase C fallback below.
            if slot_ok and slot in _BASE_RAG_SLOTS:
                _write_slot(case_dir, slot, result_data["text"])
            elif not slot_ok and fallback and slot in _BASE_RAG_SLOTS:
                logger.warning(f"RAG failed for {slot} ({result_data.get('error')}), using fallback")
                _write_slot_fallback(case_dir, slot, patches, sim_params, is_2d)
                rag_used[slot] = False
    else:
        # Pure fallback path (no RAG)
        logger.info("Using hardcoded case_builder (RAG not configured)")
        for slot in _BASE_RAG_SLOTS:
            _write_slot_fallback(case_dir, slot, patches, sim_params, is_2d)
            rag_used[slot] = False

    # ── Phase B2: write solver/scheme files via hardcoded writers ─────────────
    ALL_SOLVER_SLOTS = [
        "constant/physicalProperties",
        "constant/momentumTransport",
        "system/fvSchemes",
        "system/fvSolution",
        "system/controlDict",
    ]
    for slot in ALL_SOLVER_SLOTS:
        _write_slot_fallback(case_dir, slot, patches, sim_params, is_2d)

    # ── Phase B3: (no post-patching needed — RANS writers already include all div-schemes) ──

    # ── Phase C: turbulence IC fields (k / epsilon / omega / nut) ────────────
    if turb_model != "laminar":
        logger.info(f"Writing turbulence fields for {turb_model} (fallback first, then RAG overlay)")
        write_turbulence_fields(
            case_dir, patches, is_2d,
            U_mag=sim_params.get("U_mag", 1.0),
        )
        # Overwrite with RAG-generated content for turb slots that succeeded
        for slot in _TURB_RAG_SLOTS:
            slot_data = generated.get(slot, {})
            if slot_data.get("valid"):
                _write_slot(case_dir, slot, slot_data["text"])
                logger.info(f"  {slot}: RAG output applied")
            else:
                rag_used[slot] = False
                if slot_data:
                    logger.info(f"  {slot}: fallback used ({slot_data.get('error', 'no result')})")

    # ── Phase D: fix field/boundary type consistency and velocity values ─────
    _fix_patch_type_consistency(case_dir, patches, is_2d)
    _fix_velocity_values(case_dir, patches, sim_params, is_2d)

    # ── Phase E: structural repair — ensure every patch has a valid BC block ──
    _repair_bc_files(case_dir, patches, is_2d, sim_params)

    return {
        "case_dir":   case_dir,
        "patches":    patches,
        "is_2d":      is_2d,
        "sim_params": sim_params,
        "rag_used":   rag_used,
        "turb_model": turb_model,
        "error":      None,
    }


def _fix_patch_type_consistency(case_dir: str, patches: list, is_2d: bool):
    """
    After LLM writes 0/p and 0/U, ensure that patches whose polyMesh boundary
    type is 'empty' or 'symmetry' have matching patchField types.
    Replaces wrong types (e.g. zeroGradient, fixedValue) inline.
    """
    from simulation.case_builder import classify_patch
    import re

    # Determine which patches need special patchField types
    special = {}  # patch_name -> required patchField type string
    for p in patches:
        name = p["name"] if isinstance(p, dict) else p
        role = classify_patch(name, is_2d)
        if role == "empty":
            special[name] = "empty"
        elif role == "symmetry":
            special[name] = "symmetry"

    if not special:
        return

    for slot in ("0/U", "0/p"):
        fpath = os.path.join(case_dir, slot)
        if not os.path.exists(fpath):
            continue
        text = open(fpath).read()
        changed = False
        for patch_name, req_type in special.items():
            # Match the patch block: patch_name { ... type <something>; ... }
            # Replace the type line inside the block for this patch
            pattern = (
                r'(\b' + re.escape(patch_name) + r'\b\s*\{[^}]*?)'
                r'(type\s+\w+;)'
            )
            replacement = lambda m, rt=req_type: m.group(1) + f"type            {rt};"
            new_text, n = re.subn(pattern, replacement, text, flags=re.DOTALL)
            if n:
                text = new_text
                changed = True
                logger.info(f"  fixed {slot}/{patch_name} → type {req_type};")
        if changed:
            with open(fpath, "w") as f:
                f.write(text)


def _fix_velocity_values(case_dir: str, patches: list, sim_params: dict, is_2d: bool):
    """
    After LLM writes 0/U, ensure velocity-driven patches have the correct
    speed computed from Re (sim_params['U_mag']).  The LLM often writes
    hardcoded values like (1 0 0) regardless of the actual Reynolds number.

    Corrects:
    - velocity_inlet patches: fixedValue → (U_mag 0 0)
    - moving_wall patches (lid): fixedValue → (U_mag 0 0)
    - internalField: (0 0 0) for cavity/closed domains, (U_mag 0 0) otherwise
    """
    from simulation.case_builder import classify_patch
    import re, math

    fpath = os.path.join(case_dir, "0/U")
    if not os.path.exists(fpath):
        return

    U_mag = sim_params.get("U_mag", 1.0)
    AoA   = math.radians(sim_params.get("AoA_deg", 0.0))
    Ux    = U_mag * math.cos(AoA)
    Uy    = U_mag * math.sin(AoA)
    U_str = f"({Ux:.6g} {Uy:.6g} 0)"

    text = open(fpath).read()
    changed = False

    for p in patches:
        name = p["name"] if isinstance(p, dict) else p
        role = classify_patch(name, is_2d)
        if role not in ("velocity_inlet", "moving_wall"):
            continue

        # Replace the value line inside this patch's fixedValue block
        # Pattern: patch_name { ... value uniform (<anything>); ... }
        pat = (
            r'(\b' + re.escape(name) + r'\b\s*\{[^}]*?'
            r'type\s+fixedValue\s*;[^}]*?)'
            r'value\s+uniform\s+\([^)]*\)\s*;'
        )
        repl = lambda m, us=U_str: m.group(1) + f"value           uniform {us};"
        new_text, n = re.subn(pat, repl, text, flags=re.DOTALL)
        if n:
            text = new_text
            changed = True
            logger.info(f"  fixed 0/U/{name} velocity → {U_str}")

    # Fix internalField: cavity/closed flows start at rest; open flows at U_mag
    has_inlet = any(
        classify_patch(p["name"] if isinstance(p, dict) else p, is_2d) == "velocity_inlet"
        for p in patches
    )
    if has_inlet:
        # Open flow: initialise with inlet velocity to speed up convergence
        new_text = re.sub(
            r'internalField\s+uniform\s+\([^)]*\)',
            f'internalField   uniform {U_str}',
            text
        )
    else:
        # Closed domain (cavity): start at rest
        new_text = re.sub(
            r'internalField\s+uniform\s+\([^)]*\)',
            'internalField   uniform (0 0 0)',
            text
        )
    if new_text != text:
        text = new_text
        changed = True
        logger.info(f"  fixed 0/U internalField")

    if changed:
        with open(fpath, "w") as f:
            f.write(text)


def _bc_file_is_valid(text: str, patch_names: list) -> tuple:
    """
    Returns (ok: bool, reason: str).
    Checks: balanced braces, all patch names present as named blocks,
    and every named patch block contains a 'type' keyword.
    """
    from rag.validator import _balanced_braces, _extract_named_patches
    import re

    if not _balanced_braces(text):
        return False, "unbalanced braces"

    present = _extract_named_patches(text)
    missing = [p for p in patch_names if p not in present]
    if missing:
        return False, f"missing patches: {missing}"

    # Check each patch block has a 'type' keyword
    # Extract the boundaryField content and check per-patch block
    m = re.search(r'boundaryField\s*\{', text)
    if m:
        depth = 1
        i = m.end()
        bf_end = i
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        bf_content = text[m.end():i - 1]
        for pname in patch_names:
            # Find this patch's block
            pm = re.search(
                r'\b' + re.escape(pname) + r'\b\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                bf_content, re.DOTALL
            )
            if pm and 'type' not in pm.group(1):
                return False, f"patch '{pname}' block missing 'type' keyword"

    return True, ""


def _repair_bc_files(case_dir: str, patches: list, is_2d: bool, sim_params: dict):
    """
    Phase E: safety-net repair on 0/U and 0/p after all generation/fallback.

    Checks:
    - Balanced braces
    - All mesh patches have named blocks in boundaryField
    - Every patch block contains a 'type' keyword

    On any failure: inject missing patches first (non-destructive), then
    re-check. If still broken, replace the file with the reliable fallback writer.
    """
    from rag.validator import _extract_named_patches
    from simulation.case_builder import classify_patch

    patch_names = [p["name"] if isinstance(p, dict) else p for p in patches]

    for slot in ("0/U", "0/p"):
        fpath = os.path.join(case_dir, slot)
        if not os.path.exists(fpath):
            continue
        text = open(fpath).read()

        ok, reason = _bc_file_is_valid(text, patch_names)
        if not ok:
            # Pass 1: try non-destructive injection of missing patches
            text = inject_missing_patches(text, slot, patches, classify_patch)
            with open(fpath, "w") as f:
                f.write(text)

            # Pass 2: re-check — if still broken, use fallback writer
            ok2, reason2 = _bc_file_is_valid(text, patch_names)
            if not ok2:
                logger.warning(
                    f"  Phase E: {slot} broken ({reason}) → still broken after injection "
                    f"({reason2}) — replacing with fallback writer"
                )
                _write_slot_fallback(case_dir, slot, patches, sim_params, is_2d)
            else:
                logger.info(f"  Phase E: {slot} repaired by injection (was: {reason})")


def _write_slot(case_dir: str, slot: str, text: str):
    """Write generated text to the correct file path."""
    target = os.path.join(case_dir, slot)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w") as f:
        f.write(text)


def _write_slot_fallback(case_dir: str, slot: str, patches: list, sim_params: dict, is_2d: bool):
    """Call the corresponding hardcoded writer."""
    writer = FALLBACK_WRITERS.get(slot)
    if writer:
        writer(case_dir, patches, sim_params, is_2d)
