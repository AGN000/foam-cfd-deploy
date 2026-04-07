"""
Phase 2: OpenFOAM simulation pipeline.

Modules:
    case_builder  – builds an OpenFOAM case directory from a .msh file + prompt
    foam_runner   – runs gmshToFoam → checkMesh → foamRun
    results_viz   – post-processes results to a PNG with pyvista
"""
