"""
High-level retrieval: given a prompt + target file slot, return formatted
context strings for the LLM.
"""
import re

from .store import VectorStore
from .vectorizer import (
    vectorize_query, GEOMETRY_TYPES, TURB_MODELS, REGIMES
)


def _infer_geometry(prompt: str) -> str:
    p = prompt.lower()
    if re.search(r'naca|airfoil|aerofoil|wing', p):
        return "airfoil"
    if re.search(r'cavity|lid.?driven', p):
        return "cavity"
    if re.search(r'backward.?step|forward.?step|pitz', p):
        return "step"
    if re.search(r'cylinder', p):
        return "cylinder"
    if re.search(r't.?junction|tee.?pipe', p):
        return "junction"
    if re.search(r'pipe|channel|duct|annul', p):
        return "pipe"
    return "generic"


def _infer_turb(prompt: str, sim_params: dict = None) -> str:
    p = prompt.lower()
    Re = (sim_params or {}).get("Re")
    U  = (sim_params or {}).get("U_mag", 1.0)
    nu = (sim_params or {}).get("nu", 1e-5)
    # compute Re from params if not given
    if Re is None and U and nu:
        Re = U * 0.1 / nu  # assume L~0.1m
    if Re and Re > 4000:
        return "kepsilon"
    if re.search(r'turbul|k.epsilon|k.omega|sst', p):
        if "omega" in p:
            return "komegasst"
        return "kepsilon"
    return "laminar"


def _infer_regime(prompt: str, sim_params: dict = None) -> str:
    p = prompt.lower()
    if re.search(r'transient|unsteady|time.?dependent', p):
        return "transient"
    return "steady"


def _format_context(results: list, slot: str) -> str:
    if not results:
        return ""
    lines = []
    for r in results:
        label = f"{r['case_name']} ({r['geometry_type']}, {r['regime']}, {r['turb_model']})"
        lines.append(f"// === Example: {label} — {slot} ===")
        lines.append(r["text"].strip())
        lines.append("// ===")
        lines.append("")
    return "\n".join(lines)


class RAGRetriever:
    def __init__(self, store: VectorStore):
        self.store = store

    def retrieve_for_slot(
        self,
        prompt: str,
        file_slot: str,
        sim_params: dict = None,
        patches: list = None,
        is_2d: bool = False,
        top_k: int = 3,
    ) -> str:
        geom   = _infer_geometry(prompt)
        turb   = _infer_turb(prompt, sim_params)
        regime = _infer_regime(prompt, sim_params)

        qvec = vectorize_query(
            prompt,
            file_slot=file_slot,
            geometry_type=geom,
            turb_model=turb,
            regime=regime,
            is_2d=is_2d,
        )

        results = self.store.search(
            qvec,
            file_slot=file_slot,
            geometry_type=geom,
            turb_model=turb,
            regime=regime,
            top_k=top_k,
        )
        return _format_context(results, file_slot)

    def retrieve_top1_template(
        self,
        prompt: str,
        file_slot: str,
        sim_params: dict = None,
        patches: list = None,
        is_2d: bool = False,
    ) -> str | None:
        """Return the text of the single best-matching tutorial chunk, or None."""
        geom   = _infer_geometry(prompt)
        turb   = _infer_turb(prompt, sim_params)
        regime = _infer_regime(prompt, sim_params)
        qvec   = vectorize_query(
            prompt, file_slot=file_slot, geometry_type=geom,
            turb_model=turb, regime=regime, is_2d=is_2d,
        )
        results = self.store.search(
            qvec, file_slot=file_slot, geometry_type=geom,
            turb_model=turb, regime=regime, top_k=1,
        )
        return results[0]["text"] if results else None

    def retrieve_full_case(
        self,
        prompt: str,
        sim_params: dict = None,
        patches: list = None,
        is_2d: bool = False,
        slots: list = None,
        top_k: int = 3,
    ) -> dict:
        if slots is None:
            from .chunker import TARGET_SLOTS
            slots = TARGET_SLOTS
        return {
            slot: self.retrieve_for_slot(
                prompt, slot, sim_params, patches, is_2d, top_k
            )
            for slot in slots
        }
