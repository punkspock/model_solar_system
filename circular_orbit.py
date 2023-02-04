# simple circular orbit using Euler-Cromer method
# phi should probably be called phi here
# when i say circular coordinates i really mean polar coordinates

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass

# constants
step = 0.002  # in yrs
iterations = 1000  # test
earth_to_sun = 1.00  # in AU; in km it's 91.607
pi = np.pi  # for simplicity; in radians


@dataclass
class Point:  # we're using spherical coordinates, boys
    r: float  # range [0, 1] in AU
    phi: float  # can be [0, 2pi] -- anywhere on the unit circle


class Vector(Point):
    pass


class Sun:
    def __init__(self, pos: Point):
        self.pos = pos


class Planet:
    def __init__(self, pos: Point, vel: Vector, delta_t):
        self.pos = pos
        self.vel = vel
        self.delta_t = delta_t

    def update(self):  # Euler-Cromer method
        r = self.pos.r  # for convenience
        phi = self.pos.phi
        r_dot = self.vel.r  # r-component of polar coordinates velocity; actually remains 0
        # print('r_dot: {}'.format(r_dot))  # test
        phi_dot: float = self.vel.phi
        delta_t = self.delta_t
        factor = 4 * (pi ** 2) / (r ** 3)
        # print("Factor: {}".format(factor))  # test

        # Cartesian position components
        x = r * np.cos(phi)
        # print("x: {}".format(x))  # test
        y = r * np.sin(phi)
        # print('y: {}'.format(y))  # test
        
        # Cartesian velocity components
        # TODO: Figure out why this line is resulting in a positive number on the first step!
        vx = (r_dot * np.cos(phi)) - (r * phi_dot * np.sin(phi))  # x component; first term cancels on first step
        # print('first term of vx: {}'.format(r_dot * np.cos(phi)))  # test
        # print('second term of vx: {}'.format(r * phi_dot * np.sin(phi)))  # test
        # print('vx: {}'.format(vx))  # test
        vy = (r * phi_dot * np.cos(phi)) + (r_dot * np.sin(phi))  # y component; km/s
        # print('vy: {}'.format(vy))  # test

        # compute updated Cartesian velocity components
        vx_new = vx - (factor * x * delta_t)
        vy_new = vy - (factor * y * delta_t)

        # calculate updated Cartesian position components
        x_new = x + (vx_new * delta_t)
        # print('x_new: {}'.format(x_new))  # test
        y_new = y + (vy_new * delta_t)
        # print('y_new: {}'.format(y_new))  # test

        # update position
        self.pos.r = ((x_new ** 2) + (y_new ** 2)) ** (1 / 2)
        # self.pos.phi = phi + (phi_dot * delta_t)  # not sure about this
        self.pos.phi = np.arctan2(y_new, x_new)  # see Wikipedia page on polar coordinates

        return x_new, y_new


def main():
    # initialize the Sun at the center
    origin = Point(0, 0)
    sun = Sun(origin)  # position = origin of the plot

    # initialize the Earth
    earth_init_pos = Point(1.0, 0)  # remember this is in circular coordinates
    earth_init_vel = Vector(0.0, 2 * pi)  # no radial velocity right now; only angular in rad/s
    earth = Planet(earth_init_pos, earth_init_vel, step)

    xdata = []
    ydata = []
    times = []

    t = 0  # initialize time
    for i in range(iterations):
        x, y = earth.update()
        xdata.append(x)
        ydata.append(y)
        times.append(t)
        t += step

    # test
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(times, xdata)
    ax.plot(times, ydata)
    plt.show()

    # now animate
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 7))  # set figure size
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    earth, = ax.plot([], [], 'g.', markersize=15)
    sun = ax.plot(0, 0, 'X', markersize=30, color='yellow')
    ax.plot(xdata, ydata, 'g--')
    plt.grid(True, lw=0.3)  # plot grid
    plt.savefig('output/circular_orbit.png')

    def animate(i):
        earth.set_data(xdata[i], ydata[i])
        return earth

    animation = FuncAnimation(fig, animate, frames=iterations, interval=200, repeat=True)
    animation.save('output/circular_orbit.gif', writer='pillow')  # mp4 doesn't work
    plt.show()


if __name__ == '__main__':
    main()

