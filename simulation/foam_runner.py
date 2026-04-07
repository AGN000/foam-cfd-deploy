"""
Run an OpenFOAM 11 simulation case.

Steps
-----
1. checkMesh   – validate the polyMesh
2. foamRun     – run incompressibleFluid solver
3. parse_residuals – extract convergence history from log

Entry point
-----------
    run_simulation(case_dir, timeout=600) -> dict
"""

import os
import re
import subprocess
import logging

from .case_builder import get_of_env

logger = logging.getLogger(__name__)


# ── checkMesh ─────────────────────────────────────────────────────────────────

def check_mesh(case_dir: str) -> dict:
    """Run checkMesh and return {ok, output}."""
    env = get_of_env()
    result = subprocess.run(
        ["checkMesh", "-case", case_dir],
        capture_output=True, text=True, env=env, timeout=120,
    )
    output = result.stdout + result.stderr
    ok = "Mesh OK." in output or result.returncode == 0
    return {"ok": ok, "output": output[-3000:]}


# ── foamRun ───────────────────────────────────────────────────────────────────

def foam_run(case_dir: str, timeout: int = 600) -> dict:
    """
    Run `foamRun -solver incompressibleFluid` in case_dir.

    Returns
    -------
    dict with keys: ok, log_path, output (last 4000 chars), iterations
    """
    env = get_of_env()
    log_path = os.path.join(case_dir, "log.foamRun")

    with open(log_path, "w") as log_f:
        result = subprocess.run(
            ["foamRun", "-case", case_dir],
            stdout=log_f, stderr=subprocess.STDOUT,
            env=env, timeout=timeout,
        )

    log_text = open(log_path).read()
    ok = result.returncode == 0 and "FOAM FATAL" not in log_text

    # Count completed time steps
    iterations = len(re.findall(r'^Time = \d', log_text, re.MULTILINE))

    return {
        "ok": ok,
        "log_path": log_path,
        "output": log_text[-4000:],
        "iterations": iterations,
    }


# ── Residual parsing ──────────────────────────────────────────────────────────

def parse_residuals(case_dir: str) -> dict:
    """
    Parse residuals from the postProcessing/residuals directory
    (written by the solverInfo function object).

    Returns
    -------
    dict: {field: [list of final residuals per time step]}
    """
    res_dir = os.path.join(case_dir, "postProcessing", "residuals")
    if not os.path.isdir(res_dir):
        # Fall back to log parsing
        return _parse_residuals_from_log(case_dir)

    result = {}
    for time_dir in sorted(os.listdir(res_dir)):
        fpath = os.path.join(res_dir, time_dir, "solverInfo.dat")
        if not os.path.exists(fpath):
            continue
        lines = open(fpath).readlines()
        if len(lines) < 2:
            continue
        # Header line starts with #
        header = [h.strip() for h in lines[0].lstrip('#').split()]
        for line in lines[1:]:
            if line.startswith('#'):
                continue
            vals = line.split()
            if len(vals) < len(header):
                continue
            row = dict(zip(header, vals))
            for field in ('p_initial', 'U_0_initial', 'U_1_initial'):
                if field in row:
                    base = field.rsplit('_', 1)[0]
                    result.setdefault(base, []).append(float(row[field]))
    return result


def _parse_residuals_from_log(case_dir: str) -> dict:
    """Fallback: parse residuals from log.foamRun."""
    log_path = os.path.join(case_dir, "log.foamRun")
    if not os.path.exists(log_path):
        return {}

    log_text = open(log_path).read()
    result = {}
    # Match lines like: Solving for p, Initial residual = 0.123, ...
    for m in re.finditer(
        r'Solving for (\w+),\s+Initial residual = ([\d.eE+-]+)',
        log_text
    ):
        field = m.group(1)
        res = float(m.group(2))
        result.setdefault(field, []).append(res)
    return result


# ── Master entry point ────────────────────────────────────────────────────────

def run_simulation(case_dir: str, timeout: int = 600) -> dict:
    """
    Validate mesh and run the simulation.

    Returns
    -------
    dict with keys:
        ok           – True if simulation completed without fatal errors
        check_mesh   – checkMesh output
        iterations   – number of time steps completed
        residuals    – {field: [residual history]}
        log_path     – path to foamRun log
        output       – last ~4000 chars of solver log
        error        – error message (or None)
    """
    # 1. Check mesh
    logger.info("Running checkMesh...")
    cm = check_mesh(case_dir)
    if not cm["ok"]:
        logger.warning(f"checkMesh reported issues:\n{cm['output'][-500:]}")
        # Proceed anyway — minor issues are common with imported meshes

    # 2. Run solver
    logger.info("Running foamRun...")
    try:
        fr = foam_run(case_dir, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "check_mesh": cm["output"],
            "iterations": 0,
            "residuals": {},
            "log_path": os.path.join(case_dir, "log.foamRun"),
            "output": "",
            "error": f"Simulation timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "ok": False,
            "check_mesh": cm["output"],
            "iterations": 0,
            "residuals": {},
            "log_path": "",
            "output": "",
            "error": str(e),
        }

    # 3. Parse residuals
    residuals = parse_residuals(case_dir)

    return {
        "ok": fr["ok"],
        "check_mesh": cm["output"],
        "iterations": fr["iterations"],
        "residuals": residuals,
        "log_path": fr["log_path"],
        "output": fr["output"],
        "error": None if fr["ok"] else "Solver returned non-zero exit code",
    }
