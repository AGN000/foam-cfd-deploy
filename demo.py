"""
Interactive CFD demo: natural-language prompt → mesh → OpenFOAM simulation.

Usage:
    cd /home/ubuntu/mesh-gen-ai
    python3 demo.py "Turbulent pipe flow Re=50000, diameter 20mm, U=2.5 m/s, kOmegaSST"
    python3 demo.py   # interactive mode — prompts for input
"""
import logging
import os
import re
import sys
import tempfile
import time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logging.getLogger("rag").setLevel(logging.INFO)

from simulation.case_builder import extract_sim_params
from rag.store import VectorStore
from rag.retriever import RAGRetriever
from rag.llm_case_generator import LLMCaseGenerator
from rag.rag_case_builder import build_case_rag
from simulation.foam_runner import run_simulation


# ── Geometry detection ────────────────────────────────────────────────────────

def _detect_geometry(prompt: str) -> str:
    """Return geometry keyword from prompt."""
    p = prompt.lower()
    if any(k in p for k in ("airfoil", "aerofoil", "naca", "wing", "aerofoil")):
        return "airfoil"
    if any(k in p for k in ("cavity", "lid-driven", "lid driven")):
        return "cavity"
    if any(k in p for k in ("backward", "step")):
        return "step"
    if re.search(r'(?:flow\s+over|flow\s+around|flow\s+past|external\s+flow|crossflow|cross.flow|bluff\s+body).{0,30}cylinder'
                 r'|cylinder.{0,30}(?:flow\s+over|crossflow|cross.flow|re\s*=|reynolds)', p):
        return "cylinder"
    if any(k in p for k in ("pipe", "tube", "duct", "circular")):
        return "pipe"
    if any(k in p for k in ("channel", "planar", "plane")):
        return "channel"
    return "pipe"   # sensible default


def _parse_length(prompt: str, keywords: list, default: float) -> float:
    """Extract a length value (m) near any of the given keywords."""
    p = prompt.lower()
    for kw in keywords:
        m = re.search(rf'{kw}\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?', p)
        if m:
            val = float(m.group(1))
            unit = m.group(2) or "m"
            if unit == "mm":
                val /= 1000
            elif unit == "cm":
                val /= 100
            return val
    return default


# ── Mesh builders per geometry ────────────────────────────────────────────────

def _gmsh_run(script: str, case_dir: str):
    """Compile and run a Gmsh geo script, return .msh path or None."""
    import gmsh
    work_dir = tempfile.mkdtemp(prefix="meshgen_")
    geo_path = os.path.join(work_dir, "mesh.geo")
    msh_path = os.path.join(work_dir, "mesh.msh")
    with open(geo_path, "w") as f:
        f.write(script)
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.open(geo_path)
        gmsh.model.mesh.generate(3)
        elem_types_3d, _, _ = gmsh.model.mesh.getElements(dim=3)
        if not elem_types_3d:
            from inference.mesh_pipeline import MeshPipeline
            MeshPipeline._extrude_2d_to_3d(gmsh, span=0.01)
        gmsh.write(msh_path)
        gmsh.finalize()
        return msh_path if os.path.exists(msh_path) else None
    except Exception as e:
        try:
            gmsh.finalize()
        except Exception:
            pass
        print(f"  [mesh] gmsh error: {e}")
        return None


def make_mesh(geom: str, prompt: str, case_dir: str):
    if geom == "airfoil":
        from dataset.generators.airfoil import AirfoilGenerator
        naca = "0012"
        m = re.search(r'naca\s*(\d{4})', prompt.lower())
        if m:
            naca = m.group(1)
        aoa = 0.0
        m = re.search(r'(?:aoa|angle\s+of\s+attack|alpha)\s*[=:]?\s*(-?\d+(?:\.\d+)?)', prompt.lower())
        if m:
            aoa = float(m.group(1))
        chord = _parse_length(prompt, ["chord", "length"], 1.0)
        g = AirfoilGenerator(seed=42)
        script = g.to_gmsh_script({
            "naca": naca, "chord": chord, "angle_of_attack": aoa,
            "domain_radius": chord * 15, "wake_length": chord * 12,
            "mesh_size_far": chord * 0.65, "mesh_size_near": chord * 0.017,
            "n_points": 20, "span": chord * 0.1,
        })
        return _gmsh_run(script, case_dir)

    if geom == "cavity":
        from dataset.generators.cavity import CavityGenerator
        side = _parse_length(prompt, ["side", "length", "width", "height", "size"], 0.1)
        g = CavityGenerator(seed=42)
        script = g.to_gmsh_script({
            "variant": "2d_square", "side": side, "depth": side * 0.5,
            "mesh_size": side * 0.1,
        })
        return _gmsh_run(script, case_dir)

    if geom == "step":
        from dataset.generators.channel import ChannelGenerator
        step_h = _parse_length(prompt, ["step", "step height", "step_height"], 0.01)
        g = ChannelGenerator(seed=42)
        script = g.to_gmsh_script({
            "variant": "backward_step",
            "inlet_height": step_h * 2, "step_height": step_h,
            "width": step_h * 3,
            "inlet_length": step_h * 5, "outlet_length": step_h * 15,
            "mesh_size": step_h * 0.3,
        })
        return _gmsh_run(script, case_dir)

    if geom == "cylinder":
        from dataset.generators.cylinder import build_cylinder_mesh_api
        import tempfile
        dia_m = re.search(r'(?:diameter|diam|d)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?', prompt.lower())
        r_m   = re.search(r'radius\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)?', prompt.lower())
        if dia_m:
            val = float(dia_m.group(1))
            unit = dia_m.group(2) or "m"
            if unit == "mm": val /= 1000
            elif unit == "cm": val /= 100
            R = val / 2
        elif r_m:
            val = float(r_m.group(1))
            unit = r_m.group(2) or "m"
            if unit == "mm": val /= 1000
            elif unit == "cm": val /= 100
            R = val
        else:
            R = 0.025
        is_2d = bool(re.search(r'2[- ]?d\b|two.?d(?:im)?', prompt.lower()))
        work_dir = tempfile.mkdtemp(prefix="meshgen_")
        msh_path = os.path.join(work_dir, "mesh.msh")
        if is_2d:
            p_mesh = {
                "variant": "2d_cylinder",
                "radius": R, "domain_length": R * 30, "domain_height": R * 16,
                "cylinder_x": R * 5, "depth": R * 0.1,
                "mesh_size_far": R * 1.2, "mesh_size_near": R * 0.08,
            }
        else:
            p_mesh = {
                "variant": "3d_cylinder",
                "radius": R, "domain_length": R * 30, "domain_height": R * 16,
                "cylinder_x": R * 5, "span": R * 4,
                "mesh_size_far": R * 1.2, "mesh_size_near": R * 0.08,
            }
        ok = build_cylinder_mesh_api(p_mesh, msh_path)
        return msh_path if ok else None

    if geom == "channel":
        from dataset.generators.channel import ChannelGenerator
        h = _parse_length(prompt, ["height", "width", "diameter", "size"], 0.05)
        g = ChannelGenerator(seed=42)
        script = g.to_gmsh_script({
            "variant": "straight",
            "height": h, "width": h, "length": h * 10,
            "mesh_size": h * 0.15,
        })
        return _gmsh_run(script, case_dir)

    # default: pipe
    from dataset.generators.pipe import PipeGenerator
    radius = _parse_length(prompt, ["radius"], 0.0)
    if radius == 0.0:
        diameter = _parse_length(prompt, ["diameter", "d"], 0.02)
        radius = diameter / 2
    length = _parse_length(prompt, ["length", "l"], radius * 10)
    mesh_size = max(radius * 0.3, 0.002)
    g = PipeGenerator(seed=42)
    script = g.to_gmsh_script({
        "variant": "circular",
        "radius": radius, "length": length, "mesh_size": mesh_size,
    })
    return _gmsh_run(script, case_dir)


# ── Result display ────────────────────────────────────────────────────────────

def _bar(val, lo=1e-8, hi=1.0, width=20) -> str:
    import math
    if val <= 0:
        return "[" + "-" * width + "]"
    frac = max(0.0, min(1.0, (math.log10(val) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))))
    filled = int(frac * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def print_results(build_result: dict, sim_result: dict, elapsed: float):
    print()
    print("=" * 65)
    print("  SIMULATION RESULTS")
    print("=" * 65)

    patches = [p["name"] if isinstance(p, dict) else p for p in build_result.get("patches", [])]
    turb = build_result.get("turb_model", "laminar")
    print(f"  Geometry : {', '.join(patches)}")
    print(f"  2D       : {build_result.get('is_2d', False)}")
    print(f"  Turbulence: {turb}")
    sp = build_result.get("sim_params", {})
    print(f"  U_mag    : {sp.get('U_mag', '?')} m/s   nu={sp.get('nu', '?'):.2e}")

    print()
    print("  Boundary conditions:")
    for slot, used in build_result.get("rag_used", {}).items():
        tag = "RAG+LLM  " if used else "fallback "
        print(f"    {tag} {slot}")

    ok = sim_result.get("ok", False)
    iters = sim_result.get("iterations", 0)
    print()
    print(f"  Solver   : {'converged' if ok else 'FAILED'} after {iters} iterations")

    residuals = sim_result.get("residuals", {})
    if residuals:
        print()
        print("  Residuals (initial → final):")
        for field, vals in residuals.items():
            if vals:
                bar = _bar(vals[-1])
                print(f"    {field:12s}: {vals[0]:.2e} → {vals[-1]:.2e}  {bar}")

    if not ok and sim_result.get("error"):
        print()
        print("  Error snippet:")
        for line in sim_result["error"].splitlines()[-8:]:
            print(f"    {line}")

    print()
    print(f"  Total wall time: {elapsed:.1f}s")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_demo(prompt: str):
    print()
    print("=" * 65)
    print(f"  PROMPT: {prompt}")
    print("=" * 65)
    t0 = time.time()

    geom = _detect_geometry(prompt)
    print(f"\n  Detected geometry: {geom}")

    # 1. Mesh — save results under ~/foam_cases/<timestamp>_<geom>
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(os.path.expanduser("~"), "foam_cases")
    case_dir = os.path.join(results_root, f"{timestamp}_{geom}")
    os.makedirs(case_dir, exist_ok=True)
    print(f"  Case dir : {case_dir}")
    print("  Generating mesh...", end="", flush=True)
    mesh_path = make_mesh(geom, prompt, case_dir)
    if not mesh_path:
        print(" FAILED")
        print("  Could not generate mesh for this geometry. Exiting.")
        return
    print(f" done ({time.time()-t0:.1f}s)")

    # 2. Build OF case
    print("  Building OpenFOAM case (RAG + LLM)...")
    store = VectorStore("rag/store")
    retriever = RAGRetriever(store)
    generator = LLMCaseGenerator(base_url="http://localhost:8000/v1")

    build_result = build_case_rag(
        case_dir, mesh_path, prompt,
        rag_retriever=retriever,
        llm_generator=generator,
        fallback=True,
    )
    if build_result.get("error"):
        print(f"  CASE BUILD FAILED: {build_result['error'][:300]}")
        return

    t_build = time.time() - t0
    print(f"  Case built ({t_build:.1f}s)")

    # 3. Run simulation
    print("  Running OpenFOAM solver...", end="", flush=True)
    sim_result = run_simulation(case_dir, timeout=300)
    print(f" done")

    # 4. Display results
    print_results(build_result, sim_result, time.time() - t0)
    print(f"\n  Case files saved to: {case_dir}")


def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("CFD Simulation Demo")
        print("-------------------")
        print("Example prompts:")
        print("  Lid-driven cavity flow Re=1000, side 0.1m, U=1 m/s")
        print("  Turbulent pipe flow Re=50000, diameter 20mm, U=2.5 m/s, kOmegaSST")
        print("  NACA 0012 airfoil Re=1e6, AoA=5 deg, U=50 m/s")
        print("  Backward-facing step Re=800, step height 10mm, U=0.8 m/s")
        print()
        prompt = input("Enter simulation prompt: ").strip()
        if not prompt:
            print("No prompt given.")
            return

    run_demo(prompt)


if __name__ == "__main__":
    main()
