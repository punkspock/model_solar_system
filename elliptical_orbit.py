# Elliptical orbit using Euler-Cromer method.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import scipy.integrate as integrate

# constants and parameters
masses = []
iterations = 500  # for testing; typically is like 500-1000 for a real run
step = 0.004  # in yrs  # this is your delta_t
G = 6.6743e-11  # gravitational constant  # TODO: Find out if this needs to in AUs^3 solar masses^-1 yrs^-2
earth_mass = 3.00e-6  # in solar masses
sun_mass = 1.0  # in solar masseshttps://en.wikipedia.org/wiki/Colorado_School_of_Mines
pi = np.pi  # for convenience


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

    def __init__(self, pos: Point, vel: Vector, mass: float, delta_t: float) -> object:
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.delta_t = delta_t

    def add(self):
        masses.append(self.mass)

    def mu(self):  # method is not static, I swear
        mu = np.product(masses) / np.sum(masses)  # reduced mass

        return mu

    def cartesian_pos(self):  # coordinate conversion
        x = self.pos.r * np.cos(self.pos.theta)
        y = self.pos.r * np.sin(self.pos.theta)

        return x, y

    def cartesian_vel(self):
        r = self.pos.r
        r_dot = self.vel.r
        theta = self.pos.theta
        theta_dot = self.vel.theta

        # calculate velocities in x and y
        vx = (r_dot * np.cos(theta)) - (r * theta_dot * np.sin(theta))
        vy = (r * theta_dot * np.cos(theta)) + (r_dot * np.sin(theta))

        return vx, vy

    def update(self):  # Euler-Cromer
        # TODO: Resolve circular dependency here
        x, y = self.cartesian_pos()
        print('XY Coordinates: {}, {}'.format(x, y))  # test
        vx, vy = self.cartesian_vel()
        print('XY velocities: {}, {}'.format(vx, vy))
        r = np.sqrt((x ** 2) + (y ** 2))  # calculate radius
        factor = 4 * (np.pi ** 2) / (r ** 3)
        mu = self.mu()
        l = mu * (r ** 2) * self.vel.theta

        # calculate new x and y velocities
        vx_new = vx - (factor * x * self.delta_t)
        vy_new = vy - (factor * y * self.delta_t)

        # Euler-Cromer step
        x_new = x + vx_new * self.delta_t
        y_new = y + vy_new * self.delta_t

        # x_new, y_new = self.cartesian_pos()

        # update position in polar coordinates
        r_old = self.pos.r  # have to do this before updating position
        self.pos.r = np.sqrt((x_new ** 2) + (y_new ** 2))  # update
        r_new = self.pos.r  # which has just updated
        print('r_new: {}'.format(r_new))
        theta_dot_old = self.vel.theta  # old value
        self.vel.theta = l / (mu * r_new ** 2)  # this is always true  # new theta_dot
        self.pos.theta = self.pos.theta + (self.vel.theta * self.delta_t)
        # TODO: Update r_dot (self.vel.r) somewhere in this method
        # self.vel.r = self.vel.r - 1.0  # check out what happens when you do this lol

        return x_new, y_new


def main():
    # initialize the Sun at the center
    origin = Point(0, 0)
    sun_vel = Vector(0, 0)
    sun = Sun(origin, sun_vel, sun_mass)  # position = origin of the plot
    # the Sun doesn't actually do anything this time, it's just...there

    # initialize the Earth
    earth_init_pos = Point(1.0, 0)  # remember this is in polar coordinates
    earth_rad_vel = 1.0  # no radial velocity right now; only angular in rad/yr  # 0.0 for circular orbit
    earth_theta_vel = 3 * pi  # chosen by trial and error
    earth_init_vel = Vector(earth_rad_vel, earth_theta_vel)
    earth = Planet(earth_init_pos, earth_init_vel, earth_mass, step)
    earth.add()  # THIS MUST BE DONE

    xdata = []
    ydata = []

    t = 0  # initialize time
    for iteration in range(iterations):
        x, y = earth.update()
        xdata.append(x)
        ydata.append(y)
        t += step

    # now animate
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 7))  # set figure size
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    earth, = ax.plot([], [], 'g.', markersize=15)
    sun = ax.plot(0, 0, 'X', markersize=30, color='yellow')  # literally just a yellow mark, does nothing
    ax.plot(xdata, ydata, 'g--')
    plt.grid(True, lw=0.3)  # plot grid
    plt.savefig('output/elliptical_orbit.png')

    def animate(i):
        earth.set_data(xdata[i], ydata[i])
        return earth

    animation = FuncAnimation(fig, animate, frames=iterations, interval=200, repeat=True)
    animation.save('output/elliptical_orbit.gif', writer='pillow')  # mp4 doesn't work
    plt.show()


if __name__ == '__main__':
    main()
