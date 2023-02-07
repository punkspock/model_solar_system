# Elliptical orbit using Euler-Cromer method.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass

# constants and parameters
masses = []
iterations = 200
delta_t = 0.002  # in yrs
G = 6.6743e-11  # gravitational constant  # TODO: Find out if this needs to in AUs^3 solar masses^-1 yrs^-2

@dataclass
class Point:
    r: float
    theta: float


class Vector(Point):
    pass


class Sun:  # this is just going to go at the origin so nbd
    def __init__(self, pos: Point, vel: Vector, mass: float):
        self.pos = pos
        self.vel = vel
        self.mass = mass

    def add(self):
        masses.append(self.mass)


class Planet:  # define planet class

    def __init__(self, pos: Point, vel: Vector, mass: float):
        self.pos = pos
        self.vel = vel
        self.mass = mass

    def add(self):
        masses.append(self.mass)

    def mu(self):  # method is not static, I swear
        mu = np.product(masses) / np.sum(masses)  # reduced mass

        return mu

    def to_cartesian(self):  # coordinate conversion
        x = self.pos.r * np.cos(self.pos.theta)
        y = self.pos.r * np.sin(self.pos.theta)

        return x, y

    def effective_potential(self):
        mu = self.mu()
        r = self.pos.r
        V = 0.00  # dummy value; this should be a function

        return V

    def energy(self):  # will need this to calculate radial velocity
        r = self.pos.r
        r_dot = self.vel.r
        theta_dot = self.vel.theta
        T = 0.5 * self.mu() * ((r_dot ** 2) + (r ** 2) * (theta_dot ** 2))
        V = self.effective_potential()

    def update(self):
        x, y = self.to_cartesian()
        r = np.sqrt(x ** 2 + y ** 2)  # calculate radius
        r_new = 0.00  # dummy value
        theta_new = 0.00
        self.pos.r = r_new
        self.pos.theta = theta_new
        x_new, y_new = self.to_cartesian()

        return x_new, y_new
