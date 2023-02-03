# Simulate simple Keplerian orbit of a planet around a Sun. Uses Euler-Cromer method.
# This orbit is circular.

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation, PillowWriter
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
    def __init__(self, pos: Point, vel: Vector, delta_t: float, orbit_rad: float):
        self.pos = pos
        self.vel = vel
        self.delta_t = delta_t
        self.orbit_rad = orbit_rad

    # functions to do the actual calculations
    def calc_radius(self):
        x = self.pos.x
        y = self.pos.y
        self.orbit_rad = (x ** 2 + y ** 2) ** (1 / 2)

    def calc_factor(self):
        factor = 4.0 * (pi**2) / self.orbit_rad**3

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


# functions
def radius(x, y):
    r = (x**2 + y**2)**(1/2)
    return r


if __name__ == '__main__':
    # initialize sun and planet
    center = Point(0.0, 0.0)
    sun = Sun(center)

    earth_init_pos = Point(earth_to_sun, earth_to_sun)  # in millions of km
    earth_init_vel = Vector(0.01, 0.01)  # (millions of km)/s # need to figure out what initial velocity should be!
    earth = Planet(earth_init_pos, earth_init_vel, step, earth_to_sun)

    # lists to save the data to
    x = []
    y = []
    r = []
    theta = []  # not used yet

    # data = []
    # calculate all the data
    for i in range(iterations):
        earth.update()
        x.append(earth.pos.x)
        y.append(earth.pos.y)
        r.append(radius(earth.pos.x, earth.pos.y))
        # data.append(earth.pos)

    # test
    print('Data: ')
    print(x[0])
    print(x[1])
    print(y[0])
    print(y[1])

    # now animate
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 7))  # set figure size
    ax = plt.axes(xlim=(-100, 100), ylim=(-100, 100))  # set axis limits
    earth, = ax.plot([], [], 'g.', markersize=15)
    sun = ax.plot(sun.pos.x, sun.pos.y, 'X', markersize=30, color='yellow')
    # ax.plot(x, y, 'g-')
    plt.grid(True, lw=0.3)  # plot grid
    plt.savefig('earth_sun.png')

    def animate(i):
        earth.set_data(r[i] * np.cos(x[i]), r[i] * np.sin(y[i]))
        return earth

    animation = FuncAnimation(fig, animate, frames=len(x), interval=200, repeat=True)
    animation.save('earth_sun.gif', writer='pillow')  # mp4 doesn't work
    plt.show()








