from .pipe import PipeGenerator
from .channel import ChannelGenerator
from .cavity import CavityGenerator
from .cylinder import CylinderGenerator
from .airfoil import AirfoilGenerator

ALL_GENERATORS = [
    PipeGenerator,
    ChannelGenerator,
    CavityGenerator,
    CylinderGenerator,
    AirfoilGenerator,
]
