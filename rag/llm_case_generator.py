"""
Generate OpenFOAM case files via the vllm HTTP endpoint.

Two strategies:
  - BC slots (0/U, 0/p, etc.): template-patch approach — RAG retrieves
    best matching tutorial file as base, LLM outputs JSON patch spec,
    script applies the patch. Syntax always correct.
  - Solver/scheme slots: hardcoded fallbacks (not LLM-generated).
"""
import logging
import math
import re
import re as _re
import requests

from .prompt_assembler import assemble_prompt, DEFAULT_SLOTS
from .validator import validate_foam_file, validate_patch_coverage, strip_markdown, sanitize_control_dict, fix_dimensions
from .template_patcher import build_patch_prompt, extract_patch_json, apply_patch

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL  = "checkpoints/unified/merged"

# Slots that use template-patch (LLM outputs JSON diff, not full file)
PATCH_SLOTS = {"0/U", "0/p", "0/k", "0/epsilon", "0/omega", "0/nut", "0/nuTilda"}

# BC types valid for incompressible solvers (pimpleFoam, simpleFoam, etc.)
_INCOMPRESSIBLE_VALID = {
    "fixedValue", "zeroGradient", "noSlip", "slip",
    "symmetryPlane", "symmetry", "empty", "cyclic", "cyclicAMI",
    "wall", "inletOutlet", "outletInlet", "pressureInletOutletVelocity",
    "freestreamVelocity", "freestream",
    "uniformFixedValue", "fixedFluxPressure", "totalPressure",
    "calculated", "nutkWallFunction", "nutUSpaldingWallFunction",
    "nutkWallFunction", "omegaWallFunction", "kqRWallFunction",
    "epsilonWallFunction", "fixedMean", "turbulentIntensityKineticEnergyInlet",
    "turbulentMixingLengthDissipationRateInlet",
    "turbulentMixingLengthFrequencyInlet",
    "freestreamPressure",
    "processor", "processorCyclic",
}


class LLMCaseGenerator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.05,
        base_url: str = VLLM_BASE_URL,
    ):
        self.model_name  = model_name
        self.temperature = temperature
        self.base_url    = base_url.rstrip("/")

    def _call(self, system_msg: str, user_msg: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system",  "content": system_msg},
                {"role": "user",    "content": user_msg},
            ],
            "temperature":   self.temperature,
            "max_tokens":    max_tokens,
            "stop":          ["<|im_end|>", "<|endoftext|>"],
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=240,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    def generate_slot_patched(
        self,
        slot: str,
        prompt: str,
        template: str,
        sim_params: dict,
        patches: list,
        is_2d: bool,
    ) -> tuple:
        """
        Template-patch approach: LLM outputs JSON spec, apply to template.
        Returns (text: str, valid: bool, error: str).
        """
        system_msg, user_msg = build_patch_prompt(
            slot, template, prompt, sim_params, patches, is_2d
        )
        raw = self._call(system_msg, user_msg, max_tokens=512)
        # Strip chain-of-thought before JSON extraction
        raw_json = _re.sub(r'<thinking>.*?</thinking>', '', raw, flags=_re.DOTALL).strip()
        patch = extract_patch_json(raw_json)

        patch_names = [p["name"] if isinstance(p, dict) else p for p in patches]

        if patch is None:
            logger.warning(f"  {slot}: LLM returned unparseable JSON — falling back to hardcoded writer")
            return "", False, "LLM returned unparseable JSON"

        # --- Fix 1: reject compressible BC types (incompressible solver only) ---
        bf = patch.get("boundaryField", {})
        for pname, pdata in bf.items():
            bc_type = pdata.get("type", "")
            if bc_type and bc_type not in _INCOMPRESSIBLE_VALID:
                logger.warning(f"  {slot}: patch has compressible/invalid BC type '{bc_type}' on patch '{pname}' — falling back")
                return "", False, f"invalid incompressible BC type: {bc_type}"
            # Reject uniformFixedValue with coded uniformValue (crashes solver via segfault)
            if bc_type == "uniformFixedValue" and str(pdata.get("uniformValue", "")).strip() == "coded":
                logger.warning(f"  {slot}: patch '{pname}' has uniformFixedValue/coded — falling back")
                return "", False, "uniformFixedValue/coded not allowed"
            # Reject 'empty' BC on non-2D patches (LLM retrieved 2D template for 3D case)
            if bc_type == "empty" and not is_2d and pname.lower() not in ("front", "back", "defaultfaces", "frontandback"):
                logger.warning(f"  {slot}: patch '{pname}' has empty BC but mesh is 3D — falling back")
                return "", False, f"empty BC on non-2D patch: {pname}"

        # --- Fix 2a: 2D cases — force front/back to empty ---
        if is_2d:
            for pname in list(bf):
                if pname.lower() in ("front", "back"):
                    bf[pname] = {"type": "empty"}

        # --- Fix 2b: compute freestreamVelocity vector from sim_params ---
        if slot == "0/U":
            U_mag = float(sim_params.get("U_mag", sim_params.get("velocity", 1.0)))
            AoA   = math.radians(float(sim_params.get("AoA_deg", 0.0)))
            for pdata in bf.values():
                if pdata.get("type") == "freestreamVelocity":
                    ux = U_mag * math.cos(AoA)
                    uz = U_mag * math.sin(AoA)
                    pdata["freestreamValue"] = f"uniform ({ux:.6g} 0 {uz:.6g})"

        text = apply_patch(template, patch)

        # --- Fix 2c: remove invalid/malformed freestreamDirection lines from template residue ---
        if slot == "0/U":
            text = _re.sub(r'[ \t]*freestreamDirection[^\n]*;\n?', '', text)
        text = fix_dimensions(text, slot)

        ok, err = validate_foam_file(text, slot)
        if not ok:
            logger.warning(f"  {slot}: patched file invalid ({err}) — falling back to hardcoded writer")
            return "", False, err

        # Verify that all actual mesh patches are covered in the generated BC file
        covered, missing = validate_patch_coverage(text, patch_names)
        if not covered:
            logger.warning(f"  {slot}: missing patches {missing} in generated file — falling back")
            return "", False, f"missing patches: {missing}"

        return text, ok, err

    def generate_slot(
        self,
        slot: str,
        prompt: str,
        context: str,
        sim_params: dict,
        patches: list,
        is_2d: bool,
        prev_generated: dict = None,
        max_tokens: int = 1024,
    ) -> tuple:
        """
        Full-generation fallback (used when no template is available).
        Returns (text: str, valid: bool, error: str).
        """
        system_msg, user_msg = assemble_prompt(
            slot, prompt, context, sim_params, patches, is_2d, prev_generated
        )
        raw = self._call(system_msg, user_msg, max_tokens)
        text = strip_markdown(raw)
        if slot == "system/controlDict":
            text = sanitize_control_dict(text)
        text = fix_dimensions(text, slot)
        ok, err = validate_foam_file(text, slot)
        return text, ok, err

    def generate_case_files(
        self,
        prompt: str,
        sim_params: dict,
        patches: list,
        is_2d: bool,
        retriever,                      # RAGRetriever instance
        slots: list = None,
    ) -> dict:
        """
        Generate all file slots for one case.
        BC slots use template-patch; others fall back to full generation.
        Returns {slot: {"text": str, "valid": bool, "error": str, "method": str}}.
        """
        if slots is None:
            slots = DEFAULT_SLOTS

        results = {}
        prev_generated = {}

        for slot in slots:
            if slot in PATCH_SLOTS:
                template = retriever.retrieve_top1_template(
                    prompt, slot, sim_params, patches, is_2d
                )
                if template:
                    text, ok, err = self.generate_slot_patched(
                        slot, prompt, template, sim_params, patches, is_2d
                    )
                    method = "template-patch"
                else:
                    # No template found → full generation fallback
                    context = retriever.retrieve_for_slot(
                        prompt, slot, sim_params, patches, is_2d
                    )
                    text, ok, err = self.generate_slot(
                        slot, prompt, context, sim_params, patches, is_2d, prev_generated
                    )
                    method = "full-gen-fallback"
            else:
                context = retriever.retrieve_for_slot(
                    prompt, slot, sim_params, patches, is_2d
                )
                text, ok, err = self.generate_slot(
                    slot, prompt, context, sim_params, patches, is_2d, prev_generated
                )
                method = "full-gen"

            results[slot] = {"text": text, "valid": ok, "error": err, "method": method}
            if ok:
                prev_generated[slot] = text
            logger.info(f"  {slot} [{method}]: {'OK' if ok else 'FAIL'} {err or ''}")

        # All cross-slot fixes operate on 0/U and 0/p results
        u_res = results.get("0/U", {})
        p_res = results.get("0/p", {})

        # --- Cross-slot fix: pressureInletOutletVelocity/inletOutlet in 0/U → fixedValue p ---
        # A pure-zeroGradient pressure field has no reference → singular pressure equation.
        _PRESSURE_OUTLET_U_TYPES = {"pressureInletOutletVelocity", "inletOutlet", "outletInlet"}
        if u_res.get("valid") and p_res.get("valid"):
            u_text = u_res["text"]
            p_text = p_res["text"]
            for utype in _PRESSURE_OUTLET_U_TYPES:
                for pname in _re.findall(
                    rf'(\w+)\s*\{{[^}}]*type\s+{re.escape(utype)}\s*;[^}}]*\}}', u_text, _re.DOTALL
                ):
                    # Only replace if 0/p has zeroGradient (not already fixedValue/totalPressure)
                    p_block = _re.search(
                        rf'{re.escape(pname)}\s*\{{([^}}]*)\}}', p_text, _re.DOTALL
                    )
                    if p_block and "zeroGradient" in p_block.group(1) and "fixedValue" not in p_block.group(1):
                        p_text = _re.sub(
                            rf'({re.escape(pname)}\s*\{{)[^}}]*(}})',
                            rf'\1\n        type            fixedValue;\n        value           uniform 0;\n    \2',
                            p_text, flags=_re.DOTALL
                        )
                        logger.info(f"  0/p: set fixedValue on pressure outlet '{pname}'")
            p_res["text"] = p_text

        # --- Cross-slot fix: freestreamVelocity in 0/U → freestreamPressure in 0/p ---
        u_res = results.get("0/U", {})  # re-fetch in case replaced above
        p_res = results.get("0/p", {})
        if u_res.get("valid") and p_res.get("valid"):
            u_text = u_res["text"]
            p_text = p_res["text"]
            freestream_patches = _re.findall(
                r'(\w+)\s*\{[^}]*type\s+freestreamVelocity\s*;[^}]*\}', u_text, _re.DOTALL
            )
            if freestream_patches:
                for pname in freestream_patches:
                    p_text = _re.sub(
                        rf'({re.escape(pname)}\s*\{{)[^}}]*(}})',
                        rf'\1\n        type            freestreamPressure;\n        freestreamValue  uniform 0;\n    \2',
                        p_text, flags=_re.DOTALL
                    )
                p_res["text"] = p_text
                logger.info(f"  0/p: applied freestreamPressure for patches {freestream_patches}")
            # Ensure all freestreamPressure blocks have freestreamValue (handles template residue)
            def _add_freestream_value(m):
                block = m.group(0)
                if "freestreamValue" not in block:
                    block = block.rstrip("} \n") + "\n        freestreamValue  uniform 0;\n    }"
                return block
            p_text = _re.sub(
                r'\w+\s*\{[^}]*type\s+freestreamPressure\s*;[^}]*\}',
                _add_freestream_value, p_res["text"], flags=_re.DOTALL
            )
            p_res["text"] = p_text

        return results
