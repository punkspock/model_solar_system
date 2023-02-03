# simple circular orbit using Euler-Cromer method

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass

# constants
step = 0.002  # in yrs
iterations = 10  # test
earth_to_sun = 1.00  # in AU; in km it's 91.607
pi = np.pi  # for simplicity


@dataclass
class Point:  # we're using spherical coordinates, boys
    r: float  # range [0, 1] in AU
    theta: float  # can be [0, 2pi] -- anywhere on the unit circle


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
        theta = self.pos.theta
        r_dot = self.vel.r
        theta_dot = self.vel.theta
        delta_t = self.delta_t
        factor = 4 * (pi ** 2) / (r ** 3)
        print("Factor: {}".format(factor))  # test
        # Cartesian position components
        x = r * np.cos(theta)
        y = theta * np.sin(theta)
        # Cartesian velocity components
        vx = r * theta_dot * np.sin(theta) + r_dot * np.cos(theta)  # x component
        vy = r * theta_dot * np.cos(theta) + r_dot * np.sin(theta)

        # compute updated Cartesian velocity components
        vx_new = vx - factor * x * delta_t
        vy_new = vy - factor * y * delta_t

        # calculate updated Cartesian position components
        x_new = x + vx_new * delta_t
        y_new = y + vy_new * delta_t

        # update position
        self.pos.r = (x_new ** 2 + y_new ** 2) ** (1 / 2)
        self.pos.theta = theta + (theta_dot * delta_t)

        # update velocity (still circular coordinates)
        self.vel.r = (vx_new ** 2 + vy_new ** 2) ** (1 / 2)  # r_dot
        # not updating self.vel.theta because it is constant and independent of x and y right now
        # TODO: Check this whole method and find equations of motion

        return x_new, y_new


def main():
    # initialize the Sun at the center
    origin = Point(0, 0)
    sun = Sun(origin)  # position = origin of the plot

    # initialize the Earth
    earth_init_pos = Point(1.0, 0)  # remember this is in circular coordinates
    earth_init_vel = Vector(0.0, 1.0)  # no radial velocity right now; only angular
    earth = Planet(earth_init_pos, earth_init_vel, step)

    xdata = []
    ydata = []

    for i in range(iterations):
        x, y = earth.update()
        xdata.append(x)
        ydata.append(y)

    # now animate
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 7))  # set figure size
    plt.axes(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))  # set axis limits with a lil margin
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

