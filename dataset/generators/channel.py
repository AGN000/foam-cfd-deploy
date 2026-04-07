"""Channel / duct flow geometry generators."""
from .base import BaseGenerator


class ChannelGenerator(BaseGenerator):

    def sample_params(self) -> dict:
        variant = self._choice(["straight", "backward_step", "t_junction"])
        params = {
            "variant": variant,
            "mesh_size": self._r(0.005, 0.04),
        }

        if variant == "straight":
            params.update({
                "length": self._r(0.5, 5.0),
                "width":  self._r(0.05, 0.5),
                "height": self._r(0.05, 0.5),
            })

        elif variant == "backward_step":
            inlet_h = self._r(0.05, 0.2)
            step_h  = self._r(0.01, inlet_h * 0.8)
            params.update({
                "inlet_height":  inlet_h,
                "step_height":   step_h,
                "width":         self._r(0.05, 0.3),
                "inlet_length":  self._r(0.1, 0.5),
                "outlet_length": self._r(0.5, 3.0),
            })

        elif variant == "t_junction":
            params.update({
                "main_length":   self._r(0.5, 2.0),
                "branch_length": self._r(0.2, 1.0),
                "main_width":    self._r(0.05, 0.2),
                "branch_width":  self._r(0.03, 0.15),
                "height":        self._r(0.05, 0.2),
            })

        return params

    def to_gmsh_script(self, p: dict) -> str:
        v  = p["variant"]
        ms = p["mesh_size"]

        lines = [
            "// Gmsh mesh script — ChannelGenerator",
            'SetFactory("OpenCASCADE");',
            f"lc = {ms};",
            "",
        ]

        if v == "straight":
            L, W, H = p["length"], p["width"], p["height"]
            lines += [
                f"// Straight channel: L={L}, W={W}, H={H}",
                f"Box(1) = {{0, 0, 0, {L}, {H}, {W}}};",
                'Physical Surface("inlet",1)  = {1};',
                'Physical Surface("outlet",2) = {2};',
                'Physical Surface("walls",3)  = {3, 4, 5, 6};',
                'Physical Volume("fluid",1)   = {1};',
            ]

        elif v == "backward_step":
            ih = p["inlet_height"]
            sh = p["step_height"]
            w  = p["width"]
            il = p["inlet_length"]
            ol = p["outlet_length"]
            total_h = round(ih + sh, 6)
            lines += [
                f"// Backward-facing step: inlet_h={ih}, step_h={sh}",
                f"Box(1) = {{0, {sh}, 0, {il}, {ih}, {w}}};",
                f"Box(2) = {{{il}, 0, 0, {ol}, {total_h}, {w}}};",
                "BooleanUnion(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("inlet",1)  = {1};',
                'Physical Surface("outlet",2) = {2};',
                'Physical Surface("walls",3)  = {3, 4, 5, 6, 7, 8};',
                'Physical Volume("fluid",1)   = {3};',
            ]

        elif v == "t_junction":
            ml  = p["main_length"]
            bl  = p["branch_length"]
            mw  = p["main_width"]
            bw  = p["branch_width"]
            h   = p["height"]
            cx  = round(ml / 2 - bw / 2, 6)
            cy  = round(mw / 2, 6)
            lines += [
                f"// T-junction: main={ml}x{mw}, branch={bl}x{bw}, h={h}",
                f"Box(1) = {{0, 0, 0, {ml}, {mw}, {h}}};",
                f"Box(2) = {{{cx}, {mw}, 0, {bw}, {bl}, {h}}};",
                "BooleanUnion(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("inlet_main",1)    = {1};',
                'Physical Surface("outlet_main",2)   = {2};',
                'Physical Surface("outlet_branch",3) = {3};',
                'Physical Surface("walls",4)         = {4, 5, 6, 7, 8, 9, 10};',
                'Physical Volume("fluid",1)          = {3};',
            ]

        lines += [
            "",
            f"Mesh.CharacteristicLengthMax = {ms};",
            f"Mesh.CharacteristicLengthMin = {ms / 5:.6f};",
            "Mesh 3;",
        ]
        return "\n".join(lines)
