"""Base class for all geometry generators."""
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GeometrySpec:
    """Holds the parameters and generated script for a geometry."""
    geometry_type: str
    params: dict[str, Any]
    gmsh_script: str
    description: str = ""


class BaseGenerator(ABC):
    """Abstract base for parametric geometry generators."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def _r(self, lo: float, hi: float, decimals: int = 4) -> float:
        return round(self.rng.uniform(lo, hi), decimals)

    def _ri(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def _choice(self, options: list):
        return self.rng.choice(options)

    @abstractmethod
    def sample_params(self) -> dict[str, Any]:
        """Sample a random set of valid parameters."""

    @abstractmethod
    def to_gmsh_script(self, params: dict[str, Any]) -> str:
        """Convert parameters to a Gmsh .geo script string."""

    def generate(self) -> GeometrySpec:
        params = self.sample_params()
        script = self.to_gmsh_script(params)
        return GeometrySpec(
            geometry_type=self.__class__.__name__.replace("Generator", "").lower(),
            params=params,
            gmsh_script=script,
        )
