"""
Post-process OpenFOAM results and save a PNG using pyvista.

Entry point
-----------
    visualize_results(case_dir, output_png, prompt="") -> dict
"""

import os
import re
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


# ── Find latest time directory ────────────────────────────────────────────────

def _latest_time(case_dir: str) -> str | None:
    """Return the path to the highest-numbered time directory."""
    times = []
    for d in os.listdir(case_dir):
        try:
            t = float(d)
            if t > 0:
                times.append((t, d))
        except ValueError:
            pass
    if not times:
        return None
    return os.path.join(case_dir, sorted(times)[-1][1])


# ── pyvista reader ────────────────────────────────────────────────────────────

def _read_foam(case_dir: str):
    """Read OpenFOAM case with pyvista. Returns pyvista dataset or None."""
    try:
        import pyvista as pv
        foam_file = os.path.join(case_dir, "foam.foam")
        # pyvista needs a .foam file (can be empty) in the case dir
        if not os.path.exists(foam_file):
            open(foam_file, "w").close()
        reader = pv.OpenFOAMReader(foam_file)
        reader.set_active_time_value(reader.time_values[-1])
        mesh = reader.read()
        return mesh
    except Exception as e:
        logger.warning(f"pyvista OpenFOAM reader failed: {e}")
        return None


# ── Parse field data from raw text files ─────────────────────────────────────

def _read_field_file(time_dir: str, field: str) -> np.ndarray | None:
    """Read a scalar or vector field from a plain text OF field file."""
    fpath = os.path.join(time_dir, field)
    if not os.path.exists(fpath):
        return None
    text = open(fpath).read()
    # Find the internalField block
    m = re.search(r'internalField\s+nonuniform\s+List<\w+>\s*\n(\d+)\s*\n\((.*?)\)',
                  text, re.DOTALL)
    if not m:
        # Try uniform
        m2 = re.search(r'internalField\s+uniform\s+(\(.*?\)|\S+)', text)
        if m2:
            raw = m2.group(1).strip('()')
            vals = [float(x) for x in raw.split()]
            return np.array(vals)
        return None
    n = int(m.group(1))
    raw = m.group(2).strip()
    # Vector field: lines like "(x y z)"
    if '(' in raw:
        vecs = re.findall(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)', raw)
        return np.array([[float(v) for v in vec] for vec in vecs])
    # Scalar field
    return np.array([float(x) for x in raw.split()])


def _read_cell_centers(case_dir: str) -> np.ndarray | None:
    """Read cell centres from constant/polyMesh/points (approximate)."""
    pts_file = os.path.join(case_dir, "constant", "polyMesh", "points")
    if not os.path.exists(pts_file):
        return None
    text = open(pts_file).read()
    coords = re.findall(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)', text)
    if not coords:
        return None
    return np.array([[float(c) for c in row] for row in coords])


# ── Convergence plot ──────────────────────────────────────────────────────────

def _convergence_ax(ax, residuals: dict, n_iter: int):
    """Plot residual history on ax."""
    ax.set_facecolor('#0d1117')
    colors = {'p': '#29b6f6', 'Ux': '#ff7043', 'Uy': '#66bb6a',
              'U_0': '#ff7043', 'U_1': '#66bb6a'}
    plotted = False
    for field, vals in residuals.items():
        if not vals:
            continue
        c = colors.get(field, '#aaaaaa')
        xs = np.linspace(1, n_iter, len(vals))
        ax.semilogy(xs, vals, color=c, lw=1.2, label=field)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no residuals", transform=ax.transAxes,
                color='#888', ha='center', va='center')
    ax.set_title("Convergence", color='white', fontsize=9, pad=4)
    ax.set_xlabel("iteration", color='#555', fontsize=7.5)
    ax.set_ylabel("residual", color='#555', fontsize=7.5)
    ax.tick_params(colors='#555', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    if plotted:
        ax.legend(fontsize=6.5, facecolor='#111', labelcolor='white',
                  edgecolor='#333')
    ax.axhline(1e-4, color='#ff6b6b', lw=0.7, linestyle='--', alpha=0.6)


# ── Scatter-based field plot ──────────────────────────────────────────────────

def _field_ax(ax, pts: np.ndarray, values: np.ndarray, title: str,
              cmap: str = 'coolwarm', label: str = ''):
    """Plot a 2D scatter/pseudocolor of a scalar field."""
    ax.set_facecolor('#0d1117')
    if pts is None or values is None or len(pts) == 0:
        ax.text(0.5, 0.5, f"no {title} data", transform=ax.transAxes,
                color='#888', ha='center', va='center')
        ax.set_title(title, color='white', fontsize=9, pad=4)
        return

    # Use 2D projection (x, y)
    x, y = pts[:, 0], pts[:, 1]
    # Downsample if too many points
    if len(x) > 30000:
        idx = np.random.choice(len(x), 30000, replace=False)
        x, y, values = x[idx], y[idx], values[idx]

    sc = ax.scatter(x, y, c=values, cmap=cmap, s=1.5, linewidths=0, alpha=0.85)
    cb = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors='#777', labelsize=6.5)
    cb.set_label(label, color='#777', fontsize=7)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title, color='white', fontsize=9, pad=4)
    ax.tick_params(colors='#555', labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.set_xlabel("x [m]", color='#555', fontsize=7)
    ax.set_ylabel("y [m]", color='#555', fontsize=7)


# ── Master entry point ────────────────────────────────────────────────────────

def visualize_results(case_dir: str, output_png: str,
                      prompt: str = "", residuals: dict = None) -> dict:
    """
    Read the latest time step from an OpenFOAM case and save a 3-panel PNG:
        [velocity magnitude]  [pressure]  [convergence]

    Parameters
    ----------
    case_dir    : OpenFOAM case directory
    output_png  : output image path
    prompt      : original prompt (used as figure title)
    residuals   : pre-parsed residual dict (from foam_runner.parse_residuals)

    Returns
    -------
    dict: {ok, output_png, error}
    """
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)

    if residuals is None:
        residuals = {}

    # ── Find latest time directory ────────────────────────────────────────────
    time_dir = _latest_time(case_dir)
    U_data = p_data = pts = None
    n_iter = 500

    if time_dir:
        n_iter = int(float(os.path.basename(time_dir)))
        U_raw = _read_field_file(time_dir, "U")
        p_raw = _read_field_file(time_dir, "p")

        if U_raw is not None and U_raw.ndim == 2:
            U_mag = np.linalg.norm(U_raw, axis=1)
        elif U_raw is not None:
            U_mag = np.abs(U_raw)
        else:
            U_mag = None

        U_data = U_mag
        p_data = p_raw if (p_raw is not None and p_raw.ndim == 1) else None

        pts = _read_cell_centers(case_dir)
        # If pts come from vertices but we have cell values, downsample pts to cell count
        if pts is not None and U_data is not None and len(pts) != len(U_data):
            # Use uniform random sample of pts at cell count (approximation)
            if len(pts) > len(U_data):
                idx = np.linspace(0, len(pts) - 1, len(U_data), dtype=int)
                pts = pts[idx]
            else:
                U_data = U_data[:len(pts)]
                if p_data is not None:
                    p_data = p_data[:len(pts)]

    # ── Try pyvista for better data ───────────────────────────────────────────
    try:
        import pyvista as pv
        foam_file = os.path.join(case_dir, "foam.foam")
        if not os.path.exists(foam_file):
            open(foam_file, "w").close()
        reader = pv.OpenFOAMReader(foam_file)
        if reader.time_values:
            reader.set_active_time_value(reader.time_values[-1])
            dataset = reader.read()
            n_iter = int(reader.time_values[-1])

            # Try to get internalMesh
            mesh_block = None
            if hasattr(dataset, 'keys'):
                for k in dataset.keys():
                    if 'internal' in k.lower() or 'fluid' in k.lower():
                        mesh_block = dataset[k]
                        break
                if mesh_block is None and len(dataset.keys()) > 0:
                    mesh_block = dataset[list(dataset.keys())[0]]
            elif hasattr(dataset, 'n_cells'):
                mesh_block = dataset

            if mesh_block is not None and mesh_block.n_cells > 0:
                # Get cell centers
                centers = mesh_block.cell_centers().points
                pts = centers

                if "U" in mesh_block.array_names:
                    U_vec = mesh_block["U"]
                    U_data = np.linalg.norm(U_vec, axis=1) if U_vec.ndim == 2 else np.abs(U_vec)
                if "p" in mesh_block.array_names:
                    p_data = mesh_block["p"]
    except Exception as e:
        logger.debug(f"pyvista read failed ({e}), using text parser")

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='#0d1117')

    title = prompt[:90] + "…" if len(prompt) > 90 else prompt
    fig.suptitle(f"CFD Results  ·  {title}",
                 color='white', fontsize=11, fontweight='bold', y=1.01)

    np.random.seed(0)

    _field_ax(axes[0], pts, U_data,
              "Velocity Magnitude  |U| [m/s]", cmap='plasma', label='m/s')
    _field_ax(axes[1], pts, p_data,
              "Kinematic Pressure  p [m²/s²]", cmap='coolwarm', label='m²/s²')
    _convergence_ax(axes[2], residuals, n_iter)

    plt.tight_layout(rect=[0, 0, 1, 0.97], w_pad=3)
    plt.savefig(output_png, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close(fig)

    return {"ok": True, "output_png": output_png, "error": None}
