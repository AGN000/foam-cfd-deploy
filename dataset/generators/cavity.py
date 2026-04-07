"""Lid-driven cavity generators."""
from .base import BaseGenerator


class CavityGenerator(BaseGenerator):

    def sample_params(self) -> dict:
        variant = self._choice(["2d_square", "2d_rect", "3d_box", "stepped"])
        params = {
            "variant": variant,
            "mesh_size": self._r(0.005, 0.04),
        }

        if variant == "2d_square":
            side = self._r(0.05, 1.0)
            params.update({"side": side, "depth": self._r(0.01, 0.05)})

        elif variant == "2d_rect":
            params.update({
                "width":  self._r(0.1, 1.0),
                "height": self._r(0.05, 0.5),
                "depth":  self._r(0.01, 0.05),
            })

        elif variant == "3d_box":
            params.update({
                "width":  self._r(0.1, 1.0),
                "height": self._r(0.1, 1.0),
                "depth":  self._r(0.1, 1.0),
            })

        elif variant == "stepped":
            W = self._r(0.2, 1.0)
            H = self._r(0.1, 0.5)
            params.update({
                "width":       W,
                "height":      H,
                "step_width":  self._r(0.05, W * 0.4),
                "step_height": self._r(0.02, H * 0.4),
                "depth":       self._r(0.01, 0.05),
            })

        return params

    def to_gmsh_script(self, p: dict) -> str:
        v  = p["variant"]
        ms = p["mesh_size"]

        lines = [
            "// Gmsh mesh script — CavityGenerator",
            'SetFactory("OpenCASCADE");',
            f"lc = {ms};",
            "",
        ]

        if v == "2d_square":
            s, d = p["side"], p["depth"]
            lines += [
                f"// Lid-driven square cavity: side={s}",
                f"Box(1) = {{0, 0, 0, {s}, {s}, {d}}};",
                'Physical Surface("lid",1)   = {5};',
                'Physical Surface("walls",2) = {1, 2, 3, 4};',
                'Physical Surface("front",3) = {6};',
                'Physical Volume("fluid",1)  = {1};',
            ]

        elif v == "2d_rect":
            W, H, d = p["width"], p["height"], p["depth"]
            lines += [
                f"// Lid-driven rectangular cavity: W={W}, H={H}",
                f"Box(1) = {{0, 0, 0, {W}, {H}, {d}}};",
                'Physical Surface("lid",1)   = {5};',
                'Physical Surface("walls",2) = {1, 2, 3, 4};',
                'Physical Surface("front",3) = {6};',
                'Physical Volume("fluid",1)  = {1};',
            ]

        elif v == "3d_box":
            W, H, D = p["width"], p["height"], p["depth"]
            lines += [
                f"// 3D lid-driven cavity: W={W}, H={H}, D={D}",
                f"Box(1) = {{0, 0, 0, {W}, {H}, {D}}};",
                'Physical Surface("lid",1)   = {5};',
                'Physical Surface("walls",2) = {1, 2, 3, 4, 6};',
                'Physical Volume("fluid",1)  = {1};',
            ]

        elif v == "stepped":
            W, H  = p["width"], p["height"]
            sw, sh = p["step_width"], p["step_height"]
            d = p["depth"]
            lines += [
                f"// Stepped cavity: W={W}, H={H}, step=({sw}x{sh})",
                f"Box(1) = {{0, 0, 0, {W}, {H}, {d}}};",
                f"Box(2) = {{0, 0, 0, {sw}, {sh}, {d}}};",
                "BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("lid",1)   = {5};',
                'Physical Surface("walls",2) = {1, 2, 3, 4, 6, 7, 8};',
                'Physical Volume("fluid",1)  = {3};',
            ]

        lines += [
            "",
            f"Mesh.CharacteristicLengthMax = {ms};",
            f"Mesh.CharacteristicLengthMin = {ms / 5:.6f};",
            "Mesh 3;",
        ]
        return "\n".join(lines)
