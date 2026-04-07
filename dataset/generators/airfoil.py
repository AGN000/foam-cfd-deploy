"""NACA 4-digit airfoil generator."""
import math
from .base import BaseGenerator


def naca4_coords(digits: str, n_points: int = 50) -> tuple[list, list, list, list]:
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0

    beta = [math.pi * i / (n_points - 1) for i in range(n_points)]
    xc   = [(1 - math.cos(b)) / 2 for b in beta]

    def thickness(x):
        return 5 * t * (0.2969 * x**0.5 - 0.1260 * x
                        - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    def camber_slope(x):
        if p == 0:
            return 0.0, 0.0
        if x <= p:
            return m / p**2 * (2 * p * x - x**2), 2 * m / p**2 * (p - x)
        return m / (1-p)**2 * ((1-2*p) + 2*p*x - x**2), 2*m/(1-p)**2*(p-x)

    ux, uy, lx, ly = [], [], [], []
    for x in xc:
        yt = thickness(max(x, 1e-9))
        yc, dyc = camber_slope(x)
        theta = math.atan(dyc)
        ux.append(x - yt * math.sin(theta))
        uy.append(yc + yt * math.cos(theta))
        lx.append(x + yt * math.sin(theta))
        ly.append(yc - yt * math.cos(theta))
    return ux, uy, lx, ly


class AirfoilGenerator(BaseGenerator):

    NACA_SERIES = [
        "0006", "0008", "0010", "0012", "0015", "0018",
        "2412", "2415", "4412", "4415", "6412",
    ]

    def sample_params(self) -> dict:
        naca  = self._choice(self.NACA_SERIES)
        chord = self._r(0.1, 1.0)
        return {
            "naca":            naca,
            "chord":           chord,
            "angle_of_attack": round(self._r(-10.0, 20.0), 2),
            "domain_radius":   self._r(chord * 10, chord * 25),
            "wake_length":     self._r(chord * 8,  chord * 20),
            "mesh_size_far":   self._r(chord * 0.05, chord * 0.15),
            "mesh_size_near":  self._r(chord * 0.002, chord * 0.01),
            "n_points":        self._ri(30, 60),
            "span":            self._r(chord * 0.1, chord * 0.3),
        }

    def to_gmsh_script(self, p: dict) -> str:
        naca     = p["naca"]
        chord    = p["chord"]
        R_domain = p["domain_radius"]
        wake_l   = p["wake_length"]
        ms_far   = p["mesh_size_far"]
        ms_near  = p["mesh_size_near"]
        n_pts    = p["n_points"]
        span     = p["span"]

        ux, uy, lx, ly = naca4_coords(naca, n_pts)
        ux = [x * chord for x in ux]
        uy = [y * chord for y in uy]
        lx = [x * chord for x in lx]
        ly = [y * chord for y in ly]

        lines = [
            f"// Gmsh mesh script — NACA {naca} airfoil",
            f"// chord={chord}, AoA={p['angle_of_attack']} deg",
            'SetFactory("OpenCASCADE");',
            f"lc_far  = {ms_far};",
            f"lc_near = {ms_near};",
            "",
        ]

        # Write airfoil points
        pt = 1
        upper_ids, lower_ids = [], []

        for x, y in zip(ux, uy):
            lines.append(f"Point({pt}) = {{{x:.6f}, {y:.6f}, 0, lc_near}};")
            upper_ids.append(pt); pt += 1

        # lower surface — skip shared TE (idx 0) and LE (idx -1)
        for x, y in zip(lx[1:-1], ly[1:-1]):
            lines.append(f"Point({pt}) = {{{x:.6f}, {y:.6f}, 0, lc_near}};")
            lower_ids.append(pt); pt += 1

        te = upper_ids[0]
        le = upper_ids[-1]
        last_lower = lower_ids[-1] if lower_ids else te

        u_pts = ", ".join(str(i) for i in upper_ids)
        l_pts = ", ".join(str(i) for i in ([te] + lower_ids + [le]))

        lines += [
            "",
            f"// Upper surface: TE → LE",
            f"Spline(1) = {{{u_pts}}};",
            f"// Lower surface: TE → LE",
            f"Spline(2) = {{{l_pts}}};",
            f"// Trailing edge (TE closes the loop; TE point shared by both splines)",
            "",
        ]

        # Domain boundary points
        qc = round(chord / 4, 4)
        lines += [
            f"// C-domain boundary",
            f"Point({pt}) = {{{qc}, 0, 0, lc_far}};",
        ]
        ctr = pt; pt += 1
        lines += [f"Point({pt}) = {{{qc}, {R_domain}, 0, lc_far}};"]
        top = pt; pt += 1
        lines += [f"Point({pt}) = {{{qc}, {-R_domain}, 0, lc_far}};"]
        bot = pt; pt += 1
        lines += [f"Point({pt}) = {{{qc - R_domain}, 0, 0, lc_far}};"]
        left = pt; pt += 1
        lines += [f"Point({pt}) = {{{chord + wake_l:.4f}, {R_domain}, 0, lc_far}};"]
        wtop = pt; pt += 1
        lines += [f"Point({pt}) = {{{chord + wake_l:.4f}, {-R_domain}, 0, lc_far}};"]
        wbot = pt; pt += 1

        lines += [
            "",
            f"Circle(10) = {{{top}, {ctr}, {left}}};",
            f"Circle(11) = {{{left}, {ctr}, {bot}}};",
            f"Line(12) = {{{top}, {wtop}}};",
            f"Line(13) = {{{wtop}, {wbot}}};",
            f"Line(14) = {{{wbot}, {bot}}};",
            "",
            "Curve Loop(1) = {10, 11, -14, -13, -12};",
            "// Airfoil loop: upper (TE→LE) then reverse of lower (LE→TE)",
            "Curve Loop(2) = {1, -2};",
            "Plane Surface(1) = {1, 2};",
            "",
            'Physical Curve("farfield",1) = {10, 11, 12, 13, 14};',
            'Physical Curve("airfoil",2)  = {1, 2};',
            'Physical Surface("fluid",1)  = {1};',
            "",
            "Field[1] = Distance;",
            "Field[1].CurvesList = {1, 2};",
            "Field[2] = Threshold;",
            "Field[2].InField  = 1;",
            f"Field[2].SizeMin = {ms_near};",
            f"Field[2].SizeMax = {ms_far};",
            f"Field[2].DistMin = {chord * 0.01:.6f};",
            f"Field[2].DistMax = {chord * 0.5:.4f};",
            "Background Field = 2;",
            "",
            f"Mesh.CharacteristicLengthMax = {ms_far};",
            f"Mesh.CharacteristicLengthMin = {ms_near / 2:.6f};",
            "Mesh 2;",
        ]
        return "\n".join(lines)
