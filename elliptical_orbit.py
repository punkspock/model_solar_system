# Elliptical orbit using Euler-Cromer method.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass


# here it doesn't make sense to use polar coordinates in the same way as circular_orbit
@dataclass
class Point:
    r: float
    phi: float


class Vector(Point):
    pass


class Sun:  # this is just going to go at the origin so nbd
    def __init__(self, pos: Point, mass: float):
        self.pos = pos
        self.mass = mass


class Planet:
    def __init__(self, pos: Point, vel: Vector, mass: float, e: float):
        self.pos = pos  # position (in polar coordinates)
        self.vel = vel  # velocity (in polar coordinates)
        self.mass = mass
        self.e = e  # eccentricity

    def update(self):
        r = self.pos.r
        phi = self.pos.phi
        r_dot = self.vel.r
        phi_dot = self.vel.phi
        m = self.mass
        e = self.e

        # convert to Cartesian
        x = r * np.cos(phi)  # x component
        y = r * np.sin(phi)  # y component


if __name__ == '__main__':
    origin = Point(0, 0)
    solar_mass = 1.00  # in solar masses
    sun = Sun(origin, solar_mass)