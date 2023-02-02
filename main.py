# Simulate simple Keplerian orbit of a planet around a Sun. Uses Euler-Cromer method.
# This orbit is circular.

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
from functools import partial

# CONSTANTS
# sun_rad = 696340.00  # in km
earth_to_sun = 91.607  # in millions of km
pi = np.pi  # to save typing
step: int = 600  # in seconds  # this is what is meant later by delta_t
sec_day: int = 1440 * 60  # seconds per day
iterations = int(sec_day / step)  # iterations  # it really wants an integer


# Utilities
@dataclass
class Point:  # 2D position
    x: float
    y: float


class Vector(Point):
    pass


# Objects that go in the simulation
class Sun:
    def __init__(self, pos: Point):
        self.pos = pos  # treating the Sun as a point mass


class Planet:
    def __init__(self, pos: Point, vel: Vector, delta_t: float, rad: float):
        self.pos = pos
        self.vel = vel
        self.delta_t = delta_t
        self.rad = rad

    # functions to do the actual calculations
    def calc_radius(self):
        x = self.pos.x
        y = self.pos.y
        self.rad = (x ** 2 + y ** 2) ** (1 / 2)

    def calc_factor(self):
        factor = 4.0 * (pi**2) / self.rad**3

        return factor

    def next_step(self, vi):
        factor = self.calc_factor()
        vf = vi - (factor * self.delta_t)  # only does this for one component

        return vf

    def update(self):
        xi = self.pos.x
        yi = self.pos.y
        vxi = self.vel.x
        vyi = self.vel.y

        vxf = self.next_step(vxi)
        vyf = self.next_step(vyi)
        xf = xi + vxf * self.delta_t
        yf = yi + vyf * self.delta_t

        self.pos.x = xf
        self.pos.y = yf


if __name__ == '__main__':
    # initialize sun and planet
    center = Point(100.0, 100.0)
    sun = Sun(center)

    earth_init_pos = Point(0.0, earth_to_sun)
    earth_init_vel = Vector(1.0e3, 1.0e3)  # need to figure out what initial velocity should be!
    earth = Planet(earth_init_pos, earth_init_vel, step, earth_to_sun)

    x_list = []
    y_list = []

    # data = []
    # calculate all the data
    for i in range(iterations):
        earth.update()
        x_list.append(earth.pos.x)
        y_list.append(earth.pos.y)
        # data.append(earth.pos)

    # now animate

    # fig, ax = plt.subplots()
    # line1, = ax.plot([], [], 'ro')
    #
    #
    # def init():
    #     ax.set_xlim(0, 200)
    #     ax.set_ylim(0, 200)
    #     return line1,
    #
    #
    # def update(frame, ln, x, y):
    #     x.append(frame)
    #     y.append(np.sin(frame))
    #     ln.set_data(x, y)
    #     return ln,
    #
    #
    # ani = FuncAnimation(
    #     fig, partial(update, ln=line1, x=[], y=[]),
    #     frames=np.linspace(0, 2 * np.pi, 128),
    #     init_func=init, blit=True)
    #
    # plt.show()







