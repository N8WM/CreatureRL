"""Utility functions and classes for Mujoco environments"""

from dataclasses import dataclass

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box


# Helper classes

@dataclass
class Vector3:
    """3D float vector"""
    x: float
    y: float
    z: float


class BodyData:
    """Helper class to get body data from Mujoco environment"""
    mjenv: MujocoEnv
    name: str
    _geom = None

    def __init__(self, mjenv: MujocoEnv, name: str):
        self.name = name
        self.mjenv = mjenv

    @property
    def geom(self):
        """Get cached geom data"""
        if self._geom is None:
            self._geom = self.mjenv.data.geom(self.name)
        return self._geom

    @property
    def com(self) -> Vector3:
        """Get center of mass"""
        x, y, z = self.mjenv.get_body_com(self.name).copy()
        return Vector3(x, y, z)


# Helper functions

def adim(x: float, c: float = 1.0, d: float = 1.0) -> float:
    """
    Generate asymptotic diminishing returns for a given input
        - `c` is the asymptote
        - `d` is the rate; higher = faster approach to `c`
    """
    return c * np.tanh(d * x)
