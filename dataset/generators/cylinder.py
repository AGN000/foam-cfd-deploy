"""Cylinder / bluff body in external flow generators."""
from .base import BaseGenerator


def _classify_surfaces_by_bbox(gmsh_model, surf_tags, L, H, D, tol=1e-3):
    """
    Return a dict mapping role → [surface tags] by inspecting bounding boxes.
    Works for a box [0,L]×[0,H]×[0,D] with an arbitrary bluff-body hole.
    """
    groups = {
        "inlet": [], "outlet": [], "top": [], "bottom": [],
        "front": [], "back": [], "body": [],
    }
    for st in surf_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh_model.getBoundingBox(2, st)
        if xmax - xmin < tol and xmin < tol:
            groups["inlet"].append(st)
        elif xmax - xmin < tol and abs(xmin - L) < tol:
            groups["outlet"].append(st)
        elif ymax - ymin < tol and ymin < tol:
            groups["bottom"].append(st)
        elif ymax - ymin < tol and abs(ymin - H) < tol:
            groups["top"].append(st)
        elif zmax - zmin < tol and zmin < tol:
            groups["front"].append(st)
        elif zmax - zmin < tol and abs(zmin - D) < tol:
            groups["back"].append(st)
        else:
            groups["body"].append(st)
    return groups


def build_cylinder_mesh_api(params: dict, msh_path: str) -> bool:
    """
    Build a cylinder-in-crossflow mesh using the gmsh Python API.
    Identifies surfaces by bounding box so physical group names are always
    correct regardless of OCC tag renumbering after BooleanDifference.

    2d_cylinder: meshes the 2D cross-section then extrudes one cell in z
                 (guarantees single-layer hex mesh suitable for OpenFOAM 2D).
    3d_cylinder: full 3D BooleanDifference mesh.

    Writes MSH2 to msh_path. Returns True on success.
    """
    import gmsh
    import logging
    log = logging.getLogger(__name__)

    v       = params["variant"]
    R       = params["radius"]
    L       = params["domain_length"]
    H       = params["domain_height"]
    cx      = params["cylinder_x"]
    cy      = H / 2
    ms_far  = params["mesh_size_far"]
    ms_near = params["mesh_size_near"]

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.model.add("cylinder_flow")

        if v == "2d_cylinder":
            # ── 2D approach: cut 2D surfaces, then extrude once in z ──────────
            D = params.get("depth", R * 0.1)

            gmsh.model.occ.addRectangle(0, 0, 0, L, H)       # surf 1
            gmsh.model.occ.addDisk(cx, cy, 0, R, R)           # surf 2
            out, _ = gmsh.model.occ.cut(
                [(2, 1)], [(2, 2)], tag=3,
                removeObject=True, removeTool=True,
            )
            gmsh.model.occ.synchronize()
            surf_2d = out[0][1]   # should be 3

            # Refinement field based on 2D curves of the cylinder boundary
            # Cylinder boundary = curves not on the outer rectangle
            tol = 1e-4
            outer_curves, cyl_curves = [], []
            for _, ctag in gmsh.model.getEntities(1):
                bb = gmsh.model.getBoundingBox(1, ctag)
                x0,y0,_,x1,y1,_ = bb
                on_box = (abs(x0) < tol or abs(x0-L) < tol or
                          abs(y0) < tol or abs(y0-H) < tol or
                          abs(x1) < tol or abs(x1-L) < tol or
                          abs(y1) < tol or abs(y1-H) < tol)
                (outer_curves if on_box else cyl_curves).append(ctag)

            if cyl_curves:
                f1 = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f1, "CurvesList", cyl_curves)
                f2 = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f2, "InField",  f1)
                gmsh.model.mesh.field.setNumber(f2, "SizeMin",  ms_near)
                gmsh.model.mesh.field.setNumber(f2, "SizeMax",  ms_far)
                gmsh.model.mesh.field.setNumber(f2, "DistMin",  R)
                gmsh.model.mesh.field.setNumber(f2, "DistMax",  R * 5)
                gmsh.model.mesh.field.setAsBackgroundMesh(f2)

            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms_far)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms_near / 2)
            gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay 2D
            gmsh.model.mesh.generate(2)

            # Extrude one cell in z (recombine → quad/hex mesh)
            ext = gmsh.model.occ.extrude(
                [(2, surf_2d)], 0, 0, D,
                numElements=[1], recombine=True,
            )
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)

            # Identify extruded entities
            back_surf = ext[0][1]                              # extruded copy of surf_2d
            vol_tag   = ext[1][1]
            lat_surfs = [e[1] for e in ext[2:] if e[0] == 2]  # lateral faces

            # Classify lateral surfaces by 2D bounding box (z ignored)
            inlet, outlet, top, bottom, cyl_surfs = [], [], [], [], []
            for st in lat_surfs:
                bb = gmsh.model.getBoundingBox(2, st)
                x0,y0,_,x1,y1,_ = bb
                if abs(x1-x0) < tol and x0 < tol:
                    inlet.append(st)
                elif abs(x1-x0) < tol and abs(x0-L) < tol:
                    outlet.append(st)
                elif abs(y1-y0) < tol and y0 < tol:
                    bottom.append(st)
                elif abs(y1-y0) < tol and abs(y0-H) < tol:
                    top.append(st)
                else:
                    cyl_surfs.append(st)

            if inlet:   gmsh.model.addPhysicalGroup(2, inlet,            name="inlet")
            if outlet:  gmsh.model.addPhysicalGroup(2, outlet,           name="outlet")
            if top:     gmsh.model.addPhysicalGroup(2, top,              name="top")
            if bottom:  gmsh.model.addPhysicalGroup(2, bottom,           name="bottom")
            if cyl_surfs: gmsh.model.addPhysicalGroup(2, cyl_surfs,      name="cylinder")
            gmsh.model.addPhysicalGroup(2, [surf_2d],   name="front")
            gmsh.model.addPhysicalGroup(2, [back_surf], name="back")
            gmsh.model.addPhysicalGroup(3, [vol_tag],   name="fluid")

        elif v == "3d_cylinder":
            # ── 3D approach: BooleanDifference, classify by bbox ─────────────
            D = params["span"]

            gmsh.model.occ.addBox(0, 0, 0, L, H, D)            # vol 1
            gmsh.model.occ.addCylinder(cx, cy, 0, 0, 0, D, R)  # vol 2
            out, _ = gmsh.model.occ.cut(
                [(3, 1)], [(3, 2)], tag=3,
                removeObject=True, removeTool=True,
            )
            gmsh.model.occ.synchronize()

            vol_tag = out[0][1]
            bnd = gmsh.model.getBoundary([(3, vol_tag)], oriented=False, combined=False)
            surf_tags = [abs(s[1]) for s in bnd]

            groups = _classify_surfaces_by_bbox(gmsh.model, surf_tags, L, H, D)

            if groups["body"]:
                f1 = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", groups["body"])
                f2 = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f2, "InField",  f1)
                gmsh.model.mesh.field.setNumber(f2, "SizeMin",  ms_near)
                gmsh.model.mesh.field.setNumber(f2, "SizeMax",  ms_far)
                gmsh.model.mesh.field.setNumber(f2, "DistMin",  R)
                gmsh.model.mesh.field.setNumber(f2, "DistMax",  R * 5)
                gmsh.model.mesh.field.setAsBackgroundMesh(f2)

            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms_far)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms_near / 2)
            gmsh.model.mesh.generate(3)

            name_map = {"inlet": "inlet", "outlet": "outlet", "top": "top",
                        "bottom": "bottom", "front": "front", "back": "back",
                        "body": "cylinder"}
            for role, tags in groups.items():
                if tags:
                    gmsh.model.addPhysicalGroup(2, tags, name=name_map[role])
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

        else:
            gmsh.finalize()
            return False

        gmsh.write(msh_path)
        gmsh.finalize()
        return True

    except Exception as e:
        try:
            gmsh.finalize()
        except Exception:
            pass
        log.warning(f"build_cylinder_mesh_api failed: {e}")
        return False


class CylinderGenerator(BaseGenerator):

    def sample_params(self) -> dict:
        variant = self._choice(["2d_cylinder", "3d_cylinder", "square_cylinder", "sphere"])
        params = {
            "variant":       variant,
            "mesh_size_far":  self._r(0.04, 0.15),
            "mesh_size_near": self._r(0.002, 0.02),
        }

        if variant in ("2d_cylinder", "3d_cylinder"):
            R = self._r(0.02, 0.1)
            params.update({
                "radius":        R,
                "domain_length": self._r(R * 15, R * 40),
                "domain_height": self._r(R * 8,  R * 20),
                "cylinder_x":   self._r(R * 3,  R * 8),
            })
            if variant == "3d_cylinder":
                params["span"] = self._r(R * 2, R * 8)
            else:
                params["depth"] = R * 2   # thin 2D slab

        elif variant == "square_cylinder":
            s = self._r(0.02, 0.1)
            params.update({
                "side":          s,
                "domain_length": self._r(s * 15, s * 40),
                "domain_height": self._r(s * 8,  s * 20),
                "cylinder_x":   self._r(s * 3,  s * 8),
                "depth":         self._r(s * 2,  s * 6),
            })

        elif variant == "sphere":
            R = self._r(0.02, 0.1)
            params.update({
                "radius":        R,
                "domain_length": self._r(R * 15, R * 40),
                "domain_height": self._r(R * 8,  R * 20),
                "domain_width":  self._r(R * 8,  R * 20),
                "sphere_x":      self._r(R * 3,  R * 8),
            })

        return params

    def to_gmsh_script(self, p: dict) -> str:
        v       = p["variant"]
        ms_far  = p["mesh_size_far"]
        ms_near = p["mesh_size_near"]

        lines = [
            "// Gmsh mesh script — CylinderGenerator",
            'SetFactory("OpenCASCADE");',
            "",
        ]

        if v == "2d_cylinder":
            R  = p["radius"]
            L  = p["domain_length"]
            H  = p["domain_height"]
            cx = p["cylinder_x"]
            cy = round(H / 2, 6)
            d  = p["depth"]
            lines += [
                f"// 2D cylinder in flow: R={R}, domain={L}x{H}",
                f"Box(1) = {{0, 0, 0, {L}, {H}, {d}}};",
                f"Cylinder(2) = {{{cx}, {cy}, 0, 0, 0, {d}, {R}}};",
                "BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("inlet",1)    = {1};',
                'Physical Surface("outlet",2)   = {2};',
                'Physical Surface("top",3)      = {3};',
                'Physical Surface("bottom",4)   = {4};',
                'Physical Surface("cylinder",5) = {5};',
                'Physical Volume("fluid",1)     = {3};',
                "",
                "Field[1] = Distance;",
                "Field[1].SurfacesList = {5};",
                "Field[2] = Threshold;",
                "Field[2].InField  = 1;",
                f"Field[2].SizeMin = {ms_near};",
                f"Field[2].SizeMax = {ms_far};",
                f"Field[2].DistMin = {R};",
                f"Field[2].DistMax = {R * 5:.4f};",
                "Background Field = 2;",
            ]

        elif v == "3d_cylinder":
            R    = p["radius"]
            L    = p["domain_length"]
            H    = p["domain_height"]
            cx   = p["cylinder_x"]
            cy   = round(H / 2, 6)
            span = p["span"]
            lines += [
                f"// 3D cylinder in flow: R={R}, domain={L}x{H}x{span}",
                f"Box(1) = {{0, 0, 0, {L}, {H}, {span}}};",
                f"Cylinder(2) = {{{cx}, {cy}, 0, 0, 0, {span}, {R}}};",
                "BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("inlet",1)    = {1};',
                'Physical Surface("outlet",2)   = {2};',
                'Physical Surface("top",3)      = {3};',
                'Physical Surface("bottom",4)   = {4};',
                'Physical Surface("front",5)    = {5};',
                'Physical Surface("back",6)     = {6};',
                'Physical Surface("cylinder",7) = {7};',
                'Physical Volume("fluid",1)     = {3};',
                "",
                "Field[1] = Distance;",
                "Field[1].SurfacesList = {7};",
                "Field[2] = Threshold;",
                "Field[2].InField  = 1;",
                f"Field[2].SizeMin = {ms_near};",
                f"Field[2].SizeMax = {ms_far};",
                f"Field[2].DistMin = {R};",
                f"Field[2].DistMax = {R * 5:.4f};",
                "Background Field = 2;",
            ]

        elif v == "square_cylinder":
            s  = p["side"]
            L  = p["domain_length"]
            H  = p["domain_height"]
            cx = p["cylinder_x"]
            cy = round((H - s) / 2, 6)
            d  = p["depth"]
            lines += [
                f"// Square cylinder in flow: side={s}, domain={L}x{H}",
                f"Box(1) = {{0, 0, 0, {L}, {H}, {d}}};",
                f"Box(2) = {{{cx}, {cy}, 0, {s}, {s}, {d}}};",
                "BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("inlet",1)    = {1};',
                'Physical Surface("outlet",2)   = {2};',
                'Physical Surface("top",3)      = {3};',
                'Physical Surface("bottom",4)   = {4};',
                'Physical Surface("cylinder",5) = {5, 6, 7, 8};',
                'Physical Volume("fluid",1)     = {3};',
            ]

        elif v == "sphere":
            R  = p["radius"]
            L  = p["domain_length"]
            H  = p["domain_height"]
            W  = p["domain_width"]
            sx = p["sphere_x"]
            sy = round(H / 2, 6)
            sz = round(W / 2, 6)
            lines += [
                f"// Sphere in external flow: R={R}, domain={L}x{H}x{W}",
                f"Box(1) = {{0, 0, 0, {L}, {H}, {W}}};",
                f"Sphere(2) = {{{sx}, {sy}, {sz}, {R}}};",
                "BooleanDifference(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };",
                "",
                'Physical Surface("inlet",1)    = {1};',
                'Physical Surface("outlet",2)   = {2};',
                'Physical Surface("top",3)      = {3};',
                'Physical Surface("bottom",4)   = {4};',
                'Physical Surface("sides",5)    = {5, 6};',
                'Physical Surface("sphere",6)   = {7};',
                'Physical Volume("fluid",1)     = {3};',
                "",
                "Field[1] = Distance;",
                "Field[1].SurfacesList = {7};",
                "Field[2] = Threshold;",
                "Field[2].InField  = 1;",
                f"Field[2].SizeMin = {ms_near};",
                f"Field[2].SizeMax = {ms_far};",
                f"Field[2].DistMin = {R};",
                f"Field[2].DistMax = {R * 6:.4f};",
                "Background Field = 2;",
            ]

        lines += [
            "",
            f"Mesh.CharacteristicLengthMax = {ms_far};",
            f"Mesh.CharacteristicLengthMin = {ms_near / 2:.6f};",
            "Mesh 3;",
        ]
        return "\n".join(lines)
