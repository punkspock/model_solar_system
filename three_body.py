from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants
iterations = 500
step = 0.002  # in yr
bodies = []

@dataclass
class Point:
    r: float
    theta: float


class Vector(Point):
    pass


def center_of_mass(body_list):  # returns center of mass location in Cartesian coordinates
    mxs = []
    mys = []
    masses = []
    for body in body_list:
        mxs.append(body.mass * body.x)
        mys.append(body.mass * body.y)
        masses.append(body.mass)
    cmx = np.sum(mxs) / np.sum(masses)
    cmy = np.sum(mys) / np.sum(masses)

    return cmx, cmy


def reduced_mass(body_list):
    masses = []
    for body in body_list:
        masses.append(body.mass)
    mu = np.product(masses) / np.sum(masses)

    return mu


class Body:
    def __init__(self, pol_pos: Point, pol_vel: Vector, mass: float, delta_t: float):
        # specify
        self.r = pol_pos.r
        self.theta = pol_pos.theta
        self.r_dot = pol_vel.r
        self.theta_dot = pol_vel.theta
        self.mass = mass
        self.delta_t = delta_t

        # calculate
        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)
        self.x_dot = self.r_dot * np.cos(self.theta) - self.r * self.theta_dot * np.sin(self.theta)
        self.y_dot = self.r_dot * np.sin(self.theta) + self.r * self.theta_dot * np.cos(self.theta)

        # initialize
        self.ang_mom = 0.0
        self.x_vals = []
        self.y_vals = []

    def add(self):
        bodies.append(self)

    def calc_ang_mom(self):  # calculate ONCE after adding all bodies
        mu = reduced_mass(bodies)
        try:
            self.ang_mom = mu * (self.r ** 2) * self.theta_dot
        except (len(bodies) == 0):
            print('You haven\'t added any bodies!')
            self.ang_mom = 0.0

    def update(self):
        factor = 4 * (np.pi ** 2) / (self.r ** 3)
        mu = reduced_mass(bodies)

        # l = self.ang_mom  # why does this get bigger over time?  # test
        # print('Earth angular momentum at update() method: {}'.format(self.ang_mom))  # test

        # Euler-Cromer step
        self.x_dot = self.x_dot - (factor * self.x * self.delta_t)
        self.y_dot = self.y_dot - (factor * self.y * self.delta_t)
        self.x = self.x + (self.x_dot * self.delta_t)
        self.y = self.y + (self.y_dot * self.delta_t)

        # update position
        self.r = np.sqrt((self.x ** 2) + (self.y ** 2))
        self.theta = np.arctan2(self.y, self.x)

        # update velocity
        # notice how I update r_dot AFTER updating r
        self.r_dot = ((self.x * self.x_dot) + (self.y * self.y_dot)) / self.r
        # self.vel.theta = self.ang_mom() / (mu * (self.pos.r ** 2))  # self.ang_mom() is a constant
        self.theta_dot = self.ang_mom / (mu * (self.r ** 2))

    def run(self):
        # self.calc_ang_mom()  # want to run this method ONCE
        for iteration in range(iterations):
            self.x_vals.append(self.x)
            self.y_vals.append(self.y)
            # self.velocities.append(self.vel)
            self.update()


def main():
    # make Planet 1
    planet1_pos = Point(1.0, 0)  # radius in AU, theta in radians
    planet1_vel = Vector(0.0, 2 * np.pi)  # remember this is in Cartesian coordinates.
    planet1_mass = 3.0e-6  # in solar masses
    # earth_l = 1.88e-5  # earth initial angular momentum in solar_masses * AU**2 * rad / yr
    planet1 = Body(planet1_pos, planet1_vel, planet1_mass, step)

    # make the Sun
    planet2_pos = Point(1.0, np.pi)  # radius is in AU, theta in radians
    planet2_vel = Vector(0.0, 2 * np.pi)
    planet2_mass = 1.0  # in solar masses
    # sun_l = 0.0
    planet2 = Body(planet2_pos, planet2_vel, planet2_mass, step)

    # add all objects to the simulation at once
    planet1.add()
    planet2.add()

    # calculate angular momentum for Earth
    planet1.calc_ang_mom()  # only want to do this ONCE because angular momentum is a constant
    planet2.calc_ang_mom()
    # print('Earth angular momentum at initial calculation: {}'.format(earth.ang_mom))  # test

    # run the simulation
    for body in bodies:
        body.run()  # this only works because both Sun and Body objects have run() methods

    # now animate
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 7))  # set figure size
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    body_plots = []
    for body in bodies:
        ax.plot(body.x_vals, body.y_vals, 'y--')
        body_plot, = ax.plot([], [], 'g.', markersize=15)
        body_plots.append(body_plot)

    plt.grid(True, lw=0.3)  # plot grid
    plt.savefig('output/elliptical_orbit2.png')

    def animate(i):
        for (j, k) in zip(bodies, body_plots):
            k.set_data(j.x_vals[i], j.y_vals[i])
        return body_plots

    animation = FuncAnimation(fig, animate, frames=iterations, interval=200, repeat=True)
    animation.save('output/elliptical_orbit2.gif', writer='pillow')  # mp4 doesn't work
    plt.show()


if __name__ == '__main__':
    main()
