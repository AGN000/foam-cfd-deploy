"""
Walk OpenFOAM tutorial/working case directories and produce text chunks
(one chunk = one file slot from one case).
"""
import hashlib
import os
import re
from pathlib import Path

# File slots we care about
TARGET_SLOTS = [
    "0/U",
    "0/p",
    "0/k",
    "0/epsilon",
    "0/omega",
    "0/nuTilda",
    "constant/physicalProperties",
    "constant/momentumTransport",
    "system/fvSchemes",
    "system/fvSolution",
    "system/controlDict",
]

GEOMETRY_KEYWORDS = {
    "airfoil": ["airfoil", "aerofoil", "naca", "wing"],
    "cavity":  ["cavity", "driven", "lid"],
    "pipe":    ["pipe", "channel", "poiseuille", "couette", "duct"],
    "step":    ["step", "pitz", "backward", "forward"],
    "cylinder": ["cylinder", "cylinder"],
    "junction": ["junction", "tjunction", "tee"],
    "mixer":   ["mixer", "impeller"],
    "external": ["motorbike", "drivAer", "vehicle", "car"],
}


def _read_file(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def infer_geometry_type(case_name: str, case_dir: str = "") -> str:
    name_l = case_name.lower()
    for geom, kws in GEOMETRY_KEYWORDS.items():
        if any(k in name_l for k in kws):
            return geom
    return "generic"


def infer_regime(control_dict_text: str) -> str:
    if not control_dict_text:
        return "steady"
    if re.search(r'steadyState|SIMPLE\b', control_dict_text):
        return "steady"
    return "transient"


def infer_turb_model(mt_text: str) -> str:
    if not mt_text:
        return "laminar"
    mt_l = mt_text.lower()
    for model in ("komegasst", "komega", "kepsilon", "spalartallmaras", "nutilda",
                  "wale", "smagorinsky", "dynsmag"):
        if model in mt_l.replace("-", "").replace("_", "").replace(" ", ""):
            return model
    if "laminar" in mt_l:
        return "laminar"
    return "unknown"


def is_2d_case(case_dir: str) -> bool:
    """Heuristic: look for 'empty' patch type in 0/U or boundary file."""
    u_path = os.path.join(case_dir, "0", "U")
    if os.path.exists(u_path):
        text = _read_file(u_path)
        if "empty" in text:
            return True
    bnd = os.path.join(case_dir, "constant", "polyMesh", "boundary")
    if os.path.exists(bnd):
        if "empty" in _read_file(bnd):
            return True
    return False


def extract_file_slots(case_dir: str) -> dict:
    """Return {slot_name: content} for all existing target slots."""
    slots = {}
    for slot in TARGET_SLOTS:
        full = os.path.join(case_dir, slot)
        if os.path.exists(full):
            content = _read_file(full)
            if len(content.strip()) > 20:  # skip empty stubs
                slots[slot] = content
    return slots


def chunk_case(case_dir: str, source: str = "tutorial") -> list:
    """Return list of chunk dicts for one OpenFOAM case directory."""
    case_name = os.path.basename(case_dir)
    slots = extract_file_slots(case_dir)
    if not slots:
        return []

    geom = infer_geometry_type(case_name, case_dir)
    regime = infer_regime(slots.get("system/controlDict", ""))
    turb = infer_turb_model(slots.get("constant/momentumTransport", ""))
    two_d = is_2d_case(case_dir)

    chunks = []
    for slot, text in slots.items():
        chunk_id = hashlib.sha256(f"{case_dir}:{slot}".encode()).hexdigest()[:16]
        chunks.append({
            "id":            chunk_id,
            "text":          text,
            "file_slot":     slot,
            "case_name":     case_name,
            "geometry_type": geom,
            "turb_model":    turb,
            "regime":        regime,
            "is_2d":         two_d,
            "source":        source,
            "vector":        None,
        })
    return chunks


def walk_tutorial_cases(tutorials_root: str) -> list:
    """Walk OpenFOAM tutorials and return all chunks."""
    all_chunks = []
    root = Path(tutorials_root)

    # Focus on incompressibleFluid and fluid subdirs
    search_dirs = []
    for subdir in ["incompressibleFluid", "fluid"]:
        p = root / subdir
        if p.exists():
            search_dirs.append(p)

    for search_dir in search_dirs:
        for case_path in sorted(search_dir.iterdir()):
            if not case_path.is_dir():
                continue
            # Check it looks like a case (has system/ or 0/)
            if not any((case_path / d).exists() for d in ("system", "0")):
                continue
            chunks = chunk_case(str(case_path), source="tutorial")
            all_chunks.extend(chunks)

    return all_chunks


def walk_working_cases(case_dirs: list) -> list:
    """Index our own validated working cases."""
    all_chunks = []
    for case_dir in case_dirs:
        if os.path.exists(case_dir):
            chunks = chunk_case(case_dir, source="working")
            all_chunks.extend(chunks)
    return all_chunks
