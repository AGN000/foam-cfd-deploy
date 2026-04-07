"""
Core mesh generation pipeline with self-correction loop.

Flow:
    prompt
      └─► LLM generates Gmsh .geo script
            └─► run gmsh -3
                  ├─► SUCCESS → run gmshToFoam / convert → return mesh path
                  └─► FAIL   → append error to prompt → retry (up to max_retries)
"""
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert CFD mesh engineer. "
    "Given a natural language description of a geometry, you generate a valid Gmsh .geo script "
    "that can be run with 'gmsh script.geo -3' to produce an OpenFOAM-compatible mesh. "
    "Output only the .geo script, nothing else."
)

RETRY_SYSTEM = (
    "You are an expert CFD mesh engineer. "
    "The previous Gmsh script failed. Fix the script based on the error message. "
    "Output only the corrected .geo script, nothing else."
)

OUTPUT_DIR = "/tmp/meshgen"


class MeshPipeline:
    """Loads the fine-tuned model and runs the generation + self-correction loop."""

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self._load_model(device)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def _load_model(self, device: str):
        try:
            from vllm import LLM, SamplingParams
            logger.info("Using vllm for inference")
            self._engine = "vllm"
            self._llm = LLM(
                model=self.model_path,
                dtype="bfloat16",
                max_model_len=8192,
                gpu_memory_utilization=0.85,
            )
            self._SamplingParams = SamplingParams
        except ImportError:
            logger.info("vllm not available, falling back to transformers")
            self._load_transformers(device)

    def _load_transformers(self, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self._engine = "transformers"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self._model.eval()

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
    ) -> str:
        messages = [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
        ]

        if self._engine == "vllm":
            from vllm import SamplingParams
            tokenizer = self._llm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            outputs = self._llm.generate([prompt], params)
            return outputs[0].outputs[0].text.strip()

        else:  # transformers
            import torch
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = out[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    # ── Gmsh execution ────────────────────────────────────────────────────────

    @staticmethod
    def _fix_common_errors(script: str) -> str:
        """
        Fix common model-generated syntax errors before passing to gmsh.
        """
        # BooleanDifference/Union: model generates "} , {" (comma between blocks)
        # and "} };" (extra closing brace) instead of "} {" and "};"
        # e.g., BooleanDifference(3) = { Volume{1}; } , { Volume{2}; } };
        # → BooleanDifference(3) = { Volume{1}; } { Volume{2}; };
        if re.search(r'Boolean(?:Difference|Union|Intersection|Fragments)', script):
            script = re.sub(r'\}\s*,\s*\{', '} {', script)       # } , { → } {
            script = re.sub(r'\}\s*\}\s*;', '};', script)         # } }; → };
        return script

    @staticmethod
    def _extract_script(raw: str) -> str:
        """Strip markdown code fences if the model added them."""
        # Concatenate all fenced blocks (model may split output across multiple fences)
        blocks = re.findall(r"```(?:geo|gmsh|python)?\s*(.*?)```", raw, re.DOTALL)
        if blocks:
            return "\n".join(b.strip() for b in blocks)
        # Unclosed fence (script truncated mid-generation — strip the opening fence)
        fence_start = re.match(r"```(?:geo|gmsh|python)?\s*", raw.lstrip())
        if fence_start:
            return raw.lstrip()[fence_start.end():].strip()
        return raw.strip()

    @staticmethod
    def _run_gmsh(script: str, output_format: str = "msh2", timeout: int = 60) -> tuple[bool, str, Optional[str]]:
        """
        Run gmsh via Python API. Returns (success, msh_path_or_empty, error_msg).
        """
        import gmsh

        work_dir = tempfile.mkdtemp(prefix="meshgen_")
        geo_path = os.path.join(work_dir, "mesh.geo")
        msh_path = os.path.join(work_dir, "mesh.msh")

        with open(geo_path, "w") as f:
            f.write(script)

        if not script.strip():
            return False, "", "empty script"

        try:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # gmshToFoam requires MSH2
            gmsh.open(geo_path)
            # Scripts with "Mesh 3;" already ran meshing; calling generate(3)
            # is idempotent.  For 2D-only scripts ("Mesh 2;") generate(3) is a
            # no-op on the volume, so getElements() will find no 3D cells and
            # we fall through to the fast mesh.extrude() path.
            gmsh.model.mesh.generate(3)
            elem_types_3d, _, _ = gmsh.model.mesh.getElements(dim=3)
            if not elem_types_3d:
                # 2D-only geometry (e.g. airfoil) — extrude reusing 2D mesh
                MeshPipeline._extrude_2d_to_3d(gmsh, span=0.01)

            gmsh.write(msh_path)
            _, elem_tags, _ = gmsh.model.mesh.getElements()
            total_elements = sum(len(t) for t in elem_tags)
            gmsh.finalize()
            if not Path(msh_path).exists():
                return False, "", "gmsh wrote no output file"
            if total_elements == 0:
                return False, "", "gmsh produced no mesh elements (geometry may be incomplete)"
            return True, msh_path, None
        except Exception as e:
            try:
                gmsh.finalize()
            except Exception:
                pass
            return False, "", str(e)

    @staticmethod
    def _extrude_2d_to_3d(gmsh, span: float = 0.01):
        """
        Extrude a 2D (surface-only) Gmsh model to a thin 3D layer in-place.

        Uses gmsh.model.occ.extrude() (or geo.extrude() fallback) to create a
        3D volume from the 2D surface.  Physical Curve groups are promoted to
        Physical Surface groups so gmshToFoam assigns patches correctly.
        """
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            return

        # ── Capture physical groups before geometry changes ───────────────────
        phys_before = {}
        for dim in (1, 2):
            for _, ptag in gmsh.model.getPhysicalGroups(dim):
                name = gmsh.model.getPhysicalName(dim, ptag)
                ents = list(gmsh.model.getEntitiesForPhysicalGroup(dim, ptag))
                phys_before[(dim, name)] = ents

        curve_to_name = {}
        for (dim, name), ents in phys_before.items():
            if dim == 1:
                for ctag in ents:
                    curve_to_name[ctag] = name

        orig_surf_tags = [t for _, t in surfaces]

        # ── Collect boundary curves per surface BEFORE extrusion ─────────────
        surf_bnd_curves = {}
        for dim, stag in surfaces:
            bnd = gmsh.model.getBoundary([(dim, stag)], oriented=False, combined=False)
            surf_bnd_curves[stag] = [abs(e[1]) for e in bnd]

        # ── Extrude geometry (OCC kernel first, fall back to built-in geo) ────
        back_surfs, new_vols = [], []
        lateral_curve_surf = {}

        for dim, stag in surfaces:
            bnd_curves = surf_bnd_curves[stag]
            try:
                ext = gmsh.model.occ.extrude(
                    [(2, stag)], 0, 0, span,
                    numElements=[1], recombine=True,
                )
                gmsh.model.occ.synchronize()
            except Exception:
                ext = gmsh.model.geo.extrude(
                    [(2, stag)], 0, 0, span,
                    numElements=[1], recombine=True,
                )
                gmsh.model.geo.synchronize()

            # ext layout per surface: [(2, back), (3, vol), (2, lat0), ...]
            if len(ext) >= 2:
                back_surfs.append(ext[0][1])
                new_vols.append(ext[1][1])
                laterals = [e[1] for e in ext[2:] if e[0] == 2]
                for i, ltag in enumerate(laterals):
                    if i < len(bnd_curves):
                        lateral_curve_surf[bnd_curves[i]] = ltag

        # ── Re-mesh the new 3D volume ─────────────────────────────────────────
        gmsh.model.mesh.generate(3)

        # ── Rebuild physical groups ───────────────────────────────────────────
        gmsh.model.removePhysicalGroups()

        gmsh.model.addPhysicalGroup(2, orig_surf_tags, tag=-1)
        gmsh.model.setPhysicalName(
            2, gmsh.model.getPhysicalGroups(2)[-1][1], "front")

        if back_surfs:
            gmsh.model.addPhysicalGroup(2, back_surfs, tag=-1)
            gmsh.model.setPhysicalName(
                2, gmsh.model.getPhysicalGroups(2)[-1][1], "back")

        if new_vols:
            gmsh.model.addPhysicalGroup(3, new_vols, tag=-1)
            gmsh.model.setPhysicalName(
                3, gmsh.model.getPhysicalGroups(3)[-1][1], "fluid")

        # Promote physical curves → physical surfaces via lateral map
        name_to_surfs = {}
        for ctag, sname in curve_to_name.items():
            if ctag in lateral_curve_surf:
                name_to_surfs.setdefault(sname, []).append(lateral_curve_surf[ctag])

        for sname, stags in name_to_surfs.items():
            gmsh.model.addPhysicalGroup(2, stags, tag=-1)
            gmsh.model.setPhysicalName(
                2, gmsh.model.getPhysicalGroups(2)[-1][1], sname)

    @staticmethod
    def _convert_to_openfoam(msh_path: str, case_dir: str) -> tuple[bool, str]:
        """
        Convert .msh to OpenFOAM format using gmshToFoam.
        Returns (success, error_or_empty).
        """
        os.makedirs(case_dir, exist_ok=True)
        try:
            result = subprocess.run(
                ["gmshToFoam", msh_path, "-case", case_dir],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return True, ""
            return False, (result.stderr + result.stdout).strip()
        except FileNotFoundError:
            return False, "gmshToFoam not found — install OpenFOAM and source its environment"

    @staticmethod
    def _check_mesh(case_dir: str) -> tuple[bool, str]:
        """Run checkMesh on the OpenFOAM case."""
        try:
            result = subprocess.run(
                ["checkMesh", "-case", case_dir],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout + result.stderr
            ok = "Mesh OK." in output or result.returncode == 0
            return ok, output
        except FileNotFoundError:
            return True, "checkMesh not available (OpenFOAM not sourced)"

    # ── public API ─────────────────────────────────────────────────────────────

    @staticmethod
    def _try_deterministic_fast_path(prompt: str) -> Optional[str]:
        """
        For well-known geometry types, use the parametric generator directly
        (bypasses the LLM). Returns a Gmsh script string, or None if no match.
        """
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))

        p = prompt.lower()

        # ── NACA airfoil ────────────────────────────────────────────────────
        m = re.search(r'\bnaca\s*(\d{4})\b', prompt, re.IGNORECASE)
        if m:
            naca  = m.group(1)
            chord_m = re.search(r'chord[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            aoa_m   = re.search(r'(?:angle[_ ]of[_ ]attack|aoa)[=\s]+(-?[0-9.]+)', prompt, re.IGNORECASE)
            dom_m   = re.search(r'(?:farfield|domain|radius)[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            chord = float(chord_m.group(1)) if chord_m else 0.5
            aoa   = float(aoa_m.group(1))   if aoa_m   else 0.0
            dom   = float(dom_m.group(1))   if dom_m   else chord * 20
            try:
                from dataset.generators.airfoil import AirfoilGenerator
                gen = AirfoilGenerator(seed=42)
                return gen.to_gmsh_script({
                    "naca": naca, "chord": chord, "angle_of_attack": aoa,
                    "domain_radius": dom, "wake_length": chord * 12,
                    "mesh_size_far": chord * 0.65, "mesh_size_near": chord * 0.017,
                    "n_points": 20, "span": chord * 0.1,
                })
            except Exception:
                pass

        # ── Circular / square pipe ───────────────────────────────────────────
        if re.search(r'(?:circular|round|square|rect(?:angular)?)\s+pipe|pipe.*radius|pipe.*diam', p):
            r_m   = re.search(r'radius[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            dia_m = re.search(r'(?:diameter|diam)[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            len_m = re.search(r'length[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            R = float(r_m.group(1)) if r_m else (float(dia_m.group(1)) / 2 if dia_m else 0.05)
            L = float(len_m.group(1)) if len_m else max(1.0, R * 10)
            try:
                from dataset.generators.pipe import PipeGenerator
                gen = PipeGenerator(seed=42)
                return gen.to_gmsh_script({
                    "variant": "circular",
                    "radius": R, "length": L,
                    "mesh_size": max(R * 0.3, 0.005),
                })
            except Exception:
                pass

        # ── Lid-driven cavity ────────────────────────────────────────────────
        if re.search(r'lid.?driven|cavity', p):
            dims = re.findall(r'([0-9.]+)\s*m?\s*(?:[xX×])', prompt)
            last_m = re.search(r'([0-9.]+)\s*m\b', prompt[::-1])
            if last_m:
                dims_all = re.findall(r'([0-9.]+)\s*m?\s*[xX×]?\s*(?=[0-9]|m\b)', prompt)
            dims_all = re.findall(r'([0-9.]+)\s*(?:m\b|\s*[xX×])', prompt)
            # Try to parse NxNxN or NxN dimensions
            dims3 = re.findall(r'([0-9.]+)\s*m?\s*[xX×]\s*([0-9.]+)\s*m?\s*[xX×]\s*([0-9.]+)', prompt)
            dims2 = re.findall(r'([0-9.]+)\s*m?\s*[xX×]\s*([0-9.]+)', prompt)
            try:
                from dataset.generators.cavity import CavityGenerator
                gen = CavityGenerator(seed=42)
                if dims3:
                    W, H, D = [float(x) for x in dims3[0]]
                    return gen.to_gmsh_script({
                        "variant": "3d_box",
                        "width": W, "height": H, "depth": D,
                        "mesh_size": max(min(W, H, D) * 0.12, 0.005),
                    })
                elif dims2:
                    W, H = [float(x) for x in dims2[0]]
                    return gen.to_gmsh_script({
                        "variant": "2d_square",
                        "side": W, "depth": min(W, H) * 0.5,
                        "mesh_size": max(min(W, H) * 0.1, 0.005),
                    })
                else:
                    return gen.to_gmsh_script({
                        "variant": "3d_box",
                        "width": 0.1, "height": 0.1, "depth": 0.1,
                        "mesh_size": 0.01,
                    })
            except Exception:
                pass

        # ── Backward-facing step ─────────────────────────────────────────────
        if re.search(r'backward.?(?:facing\s+)?step', p):
            step_m   = re.search(r'step[_ ]height[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            inlet_m  = re.search(r'inlet[_ ]height[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            len_m    = re.search(r'length[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            step_h  = float(step_m.group(1))  if step_m  else 0.05
            inlet_h = float(inlet_m.group(1)) if inlet_m else step_h * 2
            length  = float(len_m.group(1))   if len_m   else 0.6
            try:
                from dataset.generators.channel import ChannelGenerator
                gen = ChannelGenerator(seed=42)
                return gen.to_gmsh_script({
                    "variant": "backward_step",
                    "inlet_height": inlet_h, "step_height": step_h,
                    "width": inlet_h * 1.5, "inlet_length": length * 0.25,
                    "outlet_length": length * 0.75, "mesh_size": step_h * 0.3,
                })
            except Exception:
                pass

        # ── Sphere in external flow ──────────────────────────────────────────
        if re.search(r'sphere|ball', p) and re.search(r'flow|over|around|past|external', p):
            r_m   = re.search(r'radius[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            dia_m = re.search(r'(?:diameter|diam)[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            R = float(r_m.group(1)) if r_m else (float(dia_m.group(1)) / 2 if dia_m else 0.05)
            try:
                from dataset.generators.cylinder import CylinderGenerator
                gen = CylinderGenerator(seed=42)
                return gen.to_gmsh_script({
                    "variant": "sphere",
                    "radius": R,
                    "domain_length": R * 30, "domain_height": R * 16, "domain_width": R * 16,
                    "sphere_x": R * 5,
                    "mesh_size_far": R * 1.2, "mesh_size_near": R * 0.1,
                })
            except Exception:
                pass

        # ── Cylinder in crossflow ────────────────────────────────────────────
        if re.search(r'cylinder.{0,30}(?:cross|flow|re\s*=|reynolds|domain)', p) or \
           re.search(r'(?:flow\s+(?:over|around|past)|cross|external).{0,20}cylinder', p):
            dia_m = re.search(r'(?:diameter|diam)[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            r_m   = re.search(r'radius[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            R = float(dia_m.group(1)) / 2 if dia_m else (float(r_m.group(1)) if r_m else 0.025)
            is_2d = bool(re.search(r'2[- ]?d\b|two.?d(?:im)?', p))
            try:
                import tempfile, os as _os
                from dataset.generators.cylinder import build_cylinder_mesh_api
                work_dir = tempfile.mkdtemp(prefix="meshgen_")
                msh_path = _os.path.join(work_dir, "mesh.msh")
                variant = "2d_cylinder" if is_2d else "3d_cylinder"
                p_mesh = {
                    "variant": variant,
                    "radius": R, "domain_length": R * 30, "domain_height": R * 16,
                    "cylinder_x": R * 5,
                    "depth" if is_2d else "span": R * 0.1 if is_2d else R * 4,
                    "mesh_size_far": R * 1.2, "mesh_size_near": R * 0.08,
                }
                if build_cylinder_mesh_api(p_mesh, msh_path):
                    return ("__prebuilt__", msh_path)
                else:
                    logger.warning("build_cylinder_mesh_api returned False — falling back to LLM")
            except Exception as _e:
                logger.warning(f"build_cylinder_mesh_api raised: {_e} — falling back to LLM")

        # ── Annular pipe ─────────────────────────────────────────────────────
        if re.search(r'annular|annulus|concentric.{0,15}pipe|pipe.{0,15}concentric', p):
            r_in_m  = re.search(r'inner[_ ]radius[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            r_out_m = re.search(r'outer[_ ]radius[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            len_m   = re.search(r'length[=\s]+([0-9.]+)\s*m?',           prompt, re.IGNORECASE)
            r_in  = float(r_in_m.group(1))  if r_in_m  else 0.02
            r_out = float(r_out_m.group(1)) if r_out_m else r_in * 2.5
            L     = float(len_m.group(1))   if len_m   else 1.0
            try:
                from dataset.generators.pipe import PipeGenerator
                gen = PipeGenerator(seed=42)
                return gen.to_gmsh_script({
                    "variant": "annular", "r_inner": r_in,
                    "r_outer": r_out, "length": L,
                    "mesh_size": r_in * 0.4,
                })
            except Exception:
                pass

        # ── T-junction channel ───────────────────────────────────────────────
        if re.search(r't[_-]?junction|t[_-]?pipe|tee.{0,10}(?:pipe|channel|junction)', p):
            mw_m = re.search(r'main[_ ]width[=\s]+([0-9.]+)\s*m?',   prompt, re.IGNORECASE)
            bw_m = re.search(r'branch[_ ]width[=\s]+([0-9.]+)\s*m?', prompt, re.IGNORECASE)
            h_m  = re.search(r'height[=\s]+([0-9.]+)\s*m?',           prompt, re.IGNORECASE)
            main_w   = float(mw_m.group(1)) if mw_m else 0.1
            branch_w = float(bw_m.group(1)) if bw_m else main_w * 0.6
            height   = float(h_m.group(1))  if h_m  else main_w * 0.5
            try:
                from dataset.generators.channel import ChannelGenerator
                gen = ChannelGenerator(seed=42)
                return gen.to_gmsh_script({
                    "variant": "t_junction",
                    "main_length": main_w * 10, "branch_length": main_w * 5,
                    "main_width": main_w, "branch_width": branch_w,
                    "height": height, "mesh_size": main_w * 0.15,
                })
            except Exception:
                pass

        return None

    def generate_script(
        self,
        prompt: str,
        max_retries: int = 3,
        temperature: float = 0.1,
        max_new_tokens: int = 4096,
    ) -> dict:
        """
        Generate a Gmsh script from a natural language prompt.
        Returns dict with keys: script, attempts, valid, error.
        """
        # Fast path: use parametric generator for known geometry types
        fast_result = self._try_deterministic_fast_path(prompt)
        if isinstance(fast_result, tuple) and fast_result[0] == "__prebuilt__":
            # API builder already wrote the mesh — validate it then return
            _, prebuilt_msh = fast_result
            return {"script": "", "attempts": 1, "valid": True, "error": None,
                    "_prebuilt_msh": prebuilt_msh}
        fast_script = fast_result
        if fast_script:
            ok, msh_path, error = self._run_gmsh(fast_script)
            if ok:
                if msh_path:
                    shutil.rmtree(os.path.dirname(msh_path), ignore_errors=True)
                return {"script": fast_script, "attempts": 1, "valid": True, "error": None}
            logger.warning(f"Fast-path gmsh failed: {error[:200]} — falling back to LLM")

        current_prompt = prompt
        system = SYSTEM_PROMPT

        for attempt in range(1, max_retries + 1):
            raw = self._call_llm(system, current_prompt, temperature, max_new_tokens)
            script = self._fix_common_errors(self._extract_script(raw))

            ok, msh_path, error = self._run_gmsh(script)
            if ok:
                # Clean up temp mesh
                if msh_path:
                    shutil.rmtree(os.path.dirname(msh_path), ignore_errors=True)
                return {"script": script, "attempts": attempt, "valid": True, "error": None}

            logger.warning(f"Attempt {attempt}/{max_retries} failed: {error[:200]}")

            # Detect truncated scripts (end mid-statement without a final Mesh directive)
            truncated = script and not re.search(r'Mesh\s+\d\s*;?\s*$', script.strip())
            system = RETRY_SYSTEM
            if truncated:
                current_prompt = (
                    f"Original request:\n{prompt}\n\n"
                    f"Your previous script was too long and got cut off before completion. "
                    f"Write a shorter, complete Gmsh .geo script. "
                    f"For airfoils, use at most 25 points per spline. "
                    f"Gmsh error:\n{error}\n\n"
                    f"Please provide a complete, concise Gmsh .geo script."
                )
            else:
                current_prompt = (
                    f"Original request:\n{prompt}\n\n"
                    f"Previous script that failed:\n```geo\n{script}\n```\n\n"
                    f"Gmsh error:\n{error}\n\n"
                    f"Please provide a corrected Gmsh .geo script."
                )

        return {"script": script, "attempts": max_retries, "valid": False, "error": error}

    def generate_mesh(
        self,
        prompt: str,
        output_format: str = "msh2",
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> dict:
        """
        Full pipeline: prompt → script → gmsh → OpenFOAM mesh.
        Returns dict with keys: script, mesh_path, attempts, valid, check_mesh_output, error.
        """
        script_result = self.generate_script(prompt, max_retries, temperature)
        if not script_result["valid"]:
            return {
                "script": script_result["script"],
                "mesh_path": "",
                "attempts": script_result["attempts"],
                "valid": False,
                "check_mesh_output": None,
                "error": script_result["error"],
            }

        script = script_result["script"]
        if "_prebuilt_msh" in script_result:
            ok, msh_path, error = True, script_result["_prebuilt_msh"], None
        else:
            ok, msh_path, error = self._run_gmsh(script, output_format)
        if not ok:
            return {
                "script": script,
                "mesh_path": "",
                "attempts": script_result["attempts"],
                "valid": False,
                "check_mesh_output": None,
                "error": error,
            }

        # Convert to OpenFOAM
        job_id = str(uuid.uuid4())[:8]
        case_dir = os.path.join(OUTPUT_DIR, f"case_{job_id}")
        conv_ok, conv_err = self._convert_to_openfoam(msh_path, case_dir)
        if not conv_ok:
            logger.warning(f"gmshToFoam failed: {conv_err} — returning raw .msh")
            # Still useful — return the .msh
            dest = os.path.join(OUTPUT_DIR, f"mesh_{job_id}.msh")
            shutil.copy(msh_path, dest)
            shutil.rmtree(os.path.dirname(msh_path), ignore_errors=True)
            return {
                "script": script,
                "mesh_path": dest,
                "attempts": script_result["attempts"],
                "valid": True,
                "check_mesh_output": None,
                "error": f"gmshToFoam unavailable: {conv_err}",
            }

        # checkMesh
        cm_ok, cm_output = self._check_mesh(case_dir)
        shutil.rmtree(os.path.dirname(msh_path), ignore_errors=True)

        return {
            "script": script,
            "mesh_path": case_dir,
            "attempts": script_result["attempts"],
            "valid": cm_ok,
            "check_mesh_output": cm_output,
            "error": None if cm_ok else "checkMesh reported errors",
        }
