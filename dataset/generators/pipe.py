"""Pipe / duct geometry generator."""
from .base import BaseGenerator


class PipeGenerator(BaseGenerator):

    def sample_params(self) -> dict:
        variant = self._choice(["circular", "square", "annular"])
        params = {
            "variant": variant,
            "length": self._r(0.1, 5.0),
            "mesh_size": self._r(0.005, 0.05),
        }
        if variant == "circular":
            params["radius"] = self._r(0.01, 0.5)
        elif variant == "square":
            params["width"]  = self._r(0.02, 0.5)
            params["height"] = self._r(0.02, 0.5)
        elif variant == "annular":
            r_inner = self._r(0.01, 0.2)
            params["r_inner"] = r_inner
            params["r_outer"] = r_inner + self._r(0.01, 0.15)
        return params

    def to_gmsh_script(self, p: dict) -> str:
        ms = p["mesh_size"]
        L  = p["length"]
        v  = p["variant"]

        lines = [
            "// Gmsh mesh script — PipeGenerator",
            'SetFactory("OpenCASCADE");',
            f"lc = {ms};",
            "",
        ]

        if v == "circular":
            R = p["radius"]
            lines += [
                f"// Circular pipe: R={R}, L={L}",
                f"Cylinder(1) = {{0, 0, 0, 0, 0, {L}, {R}}};",
                "",
                'Physical Surface("inlet",1)  = {1};',
                'Physical Surface("outlet",2) = {2};',
                'Physical Surface("wall",3)   = {3};',
                'Physical Volume("fluid",1)   = {1};',
            ]

        elif v == "square":
            W = p["width"]
            H = p["height"]
            lines += [
                f"// Square duct: W={W}, H={H}, L={L}",
                f"Box(1) = {{0, 0, 0, {W}, {H}, {L}}};",
                "",
                'Physical Surface("inlet",1)  = {1};',
                'Physical Surface("outlet",2) = {2};',
                'Physical Surface("walls",3)  = {3, 4, 5, 6};',
                'Physical Volume("fluid",1)   = {1};',
            ]

        elif v == "annular":
            ri = p["r_inner"]
            ro = p["r_outer"]
            lines += [
                f"// Annular pipe: r_inner={ri}, r_outer={ro}, L={L}",
                f"Cylinder(1) = {{0, 0, 0, 0, 0, {L}, {ro}}};",
                f"Cylinder(2) = {{0, 0, 0, 0, 0, {L}, {ri}}};",
                "BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("inlet",1)      = {1};',
                'Physical Surface("outlet",2)     = {2};',
                'Physical Surface("outer_wall",3) = {3};',
                'Physical Surface("inner_wall",4) = {4};',
                'Physical Volume("fluid",1)       = {3};',
            ]

        lines += [
            "",
            f"Mesh.CharacteristicLengthMax = {ms};",
            f"Mesh.CharacteristicLengthMin = {ms / 5:.6f};",
            "Mesh 3;",
        ]
        return "\n".join(lines)
