"""
Hash-based bag-of-words vectorizer for OpenFOAM chunks.
No external ML library required — pure numpy.
"""
import re
import numpy as np

# ── Domain vocabulary for hashing ────────────────────────────────────────────
# These tokens get reliable hash slots; other tokens still contribute via
# the hash trick but won't have guaranteed positions.
DOMAIN_VOCAB = [
    # turbulence models
    "kEpsilon", "kOmega", "kOmegaSST", "SpalartAllmaras", "nuTilda",
    "laminar", "WALE", "Smagorinsky", "LES", "RAS",
    # solvers / algorithms
    "SIMPLE", "PIMPLE", "PISO", "icoFoam", "simpleFoam", "pisoFoam",
    "incompressibleFluid", "foamRun",
    # linear solvers
    "GAMG", "smoothSolver", "PBiCGStab", "PCG", "GaussSeidel",
    "DICGaussSeidel", "DILU",
    # schemes
    "steadyState", "Euler", "backward", "CrankNicolson",
    "linearUpwind", "limitedLinear", "vanLeer", "MUSCL",
    "Gauss", "cellLimited", "faceLimited",
    # BC types
    "fixedValue", "zeroGradient", "noSlip", "freestream",
    "freestreamVelocity", "freestreamPressure",
    "inletOutlet", "outletInlet", "pressureInletOutletVelocity",
    "fixedFluxPressure", "totalPressure", "empty", "symmetry",
    "cyclic", "cyclicAMI", "wall", "movingWallVelocity",
    "turbulentIntensityKineticEnergyInlet",
    "turbulentMixingLengthFrequencyInlet",
    "turbulentMixingLengthDissipationRateInlet",
    "calculated", "slip",
    # geometry / flow keywords
    "airfoil", "aerofoil", "naca", "cavity", "pipe", "channel",
    "step", "cylinder", "junction", "duct", "mixer",
    "inlet", "outlet", "farfield", "freestream", "lid",
    # OF file classes
    "volVectorField", "volScalarField", "dictionary",
    # OF field names
    "epsilon", "omega", "nut", "nuTilda", "alphat",
    # convergence
    "residualControl", "nNonOrthogonalCorrectors", "nCorrectors",
    "relaxationFactors",
]

# ── Feature vector dimensions ─────────────────────────────────────────────────
BOW_DIM = 512
GEOMETRY_TYPES = ["airfoil", "cavity", "pipe", "step", "cylinder", "junction", "mixer", "generic"]
TURB_MODELS    = ["laminar", "kepsilon", "komega", "komegasst", "spalartallmaras", "unknown"]
REGIMES        = ["steady", "transient"]
SLOT_NAMES     = [
    "0/U", "0/p", "0/k", "0/epsilon", "0/omega", "0/nuTilda",
    "constant/physicalProperties", "constant/momentumTransport",
    "system/fvSchemes", "system/fvSolution", "system/controlDict",
]
SOURCES        = ["tutorial", "working"]

META_DIM = len(GEOMETRY_TYPES) + len(TURB_MODELS) + len(REGIMES) + len(SLOT_NAMES) + len(SOURCES) + 1  # +1 for is_2d
TOTAL_DIM = BOW_DIM + META_DIM

META_WEIGHT = 4.0  # how much metadata features are boosted relative to BOW


def _tokenize(text: str) -> list:
    return re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text)


def _dhash(tok: str, dim: int) -> int:
    """Deterministic hash using FNV-1a (no PYTHONHASHSEED dependence)."""
    h = 2166136261
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h % dim


def text_to_bow(text: str, dim: int = BOW_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokenize(text):
        for t in (tok, tok.lower()):
            vec[_dhash(t, dim)] += 1.0
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec /= nrm
    return vec


def _one_hot(value: str, options: list) -> np.ndarray:
    v = np.zeros(len(options), dtype=np.float32)
    key = value.lower().replace("-", "").replace("_", "")
    for i, opt in enumerate(options):
        if opt.lower().replace("-", "").replace("_", "") == key:
            v[i] = 1.0
            return v
    return v  # all zeros if not found


def metadata_to_vec(chunk: dict) -> np.ndarray:
    parts = [
        _one_hot(chunk.get("geometry_type", "generic"), GEOMETRY_TYPES),
        _one_hot(chunk.get("turb_model",    "unknown"),  TURB_MODELS),
        _one_hot(chunk.get("regime",         "steady"),  REGIMES),
        _one_hot(chunk.get("file_slot",      ""),        SLOT_NAMES),
        _one_hot(chunk.get("source",         "tutorial"),SOURCES),
        np.array([1.0 if chunk.get("is_2d") else 0.0], dtype=np.float32),
    ]
    v = np.concatenate(parts)
    return v * META_WEIGHT


def vectorize(chunk: dict) -> np.ndarray:
    bow = text_to_bow(chunk["text"])
    meta = metadata_to_vec(chunk)
    combined = np.concatenate([bow, meta])
    nrm = np.linalg.norm(combined)
    return combined / nrm if nrm > 0 else combined


def vectorize_query(prompt: str, file_slot: str = "",
                    geometry_type: str = "", turb_model: str = "",
                    regime: str = "steady", is_2d: bool = False) -> np.ndarray:
    fake_chunk = {
        "text":          prompt,
        "file_slot":     file_slot,
        "geometry_type": geometry_type,
        "turb_model":    turb_model,
        "regime":        regime,
        "is_2d":         is_2d,
        "source":        "tutorial",  # neutral
    }
    return vectorize(fake_chunk)
