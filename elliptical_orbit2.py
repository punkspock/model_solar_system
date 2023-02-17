from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# constants
iterations = 5000
step = 0.002  # in yrs. This will be our delta_t
bodies = []  # initialize


@dataclass
class Point:
    r: float
    theta: float


class Vector(Point):
    pass


def reduced_mass(body_list):
    masses = []
    for body in body_list:
        masses.append(body.mass)
    mu = np.product(masses) / np.sum(masses)

    return mu


# def center_of_mass(body_list):
#     masses = []  # initialize
#     positions_times_masses = []  # initialize
#     for body in body_list:
#         masses.append(body.mass)
#         pm =
class Sun:
    def __init__(self, pos: Point, mass: float):
        self.pos = pos
        self.mass = mass
        self.x_vals = []
        self.y_vals = []

    def add(self):
        bodies.append(self)

    def cartesian_pos(self):
        x = self.pos.r * np.cos(self.pos.theta)
        y = self.pos.r * np.sin(self.pos.theta)

        return x, y

    def run(self):
        for iteration in range(iterations):
            x, y = self.cartesian_pos()
            self.x_vals.append(x)
            self.y_vals.append(y)


class Body:
    def __init__(self, pos: Point, vel: Vector, mass: float, delta_t: float):
        self.pos = pos  # initial position in polar coordinates
        self.vel = vel  # initial velocity in polar coordinates
        self.mass = mass
        self.ang_mom = 0.0  # initialize; this is NOT the initial angular momentum though
        self.delta_t = delta_t  # this will actually be the same for all bodies.

        # # for convenience when i need to use Cartesian coordinates
        # self.x = self.pos.r * np.cos(self.pos.theta)
        # self.y = self.pos.r * np.sin(self.pos.theta)
        # self.vx = self.vel.r * np.cos(self.pos.theta) - self.pos.r * self.vel.theta * np.sin(self.pos.theta)
        # self.vy = self.vel
        # self.r_dot =
        # self.theta_dot = self.ang_mom / (reduced_mass(bodies) * self.radius ** 2)  # angular velocity

        # for recording
        self.x_vals = []  # initialize
        self.y_vals = []
        # self.velocities = []

    def add(self):  # gotta do this whenever you add a new body to the system
        bodies.append(self)

    def calc_ang_mom(self):  # only use this and following methods AFTER adding bodies to body list
        try:
            self.ang_mom = reduced_mass(bodies) * (self.pos.r ** 2) * self.vel.theta
        except (len(bodies) == 0):
            print('You haven\'t added any bodies yet!')
            self.ang_mom = 0.0

    def cartesian_pos(self):
        x = self.pos.r * np.cos(self.pos.theta)
        y = self.pos.r * np.sin(self.pos.theta)

        return x, y

    def cartesian_vel(self):
        vx = (self.vel.r * np.cos(self.pos.theta)) - (self.pos.r * self.vel.theta * np.sin(self.pos.theta))
        vy = (self.vel.r * np.sin(self.pos.theta)) + (self.pos.r * self.vel.theta * np.cos(self.pos.theta))

        return vx, vy

    def update(self):
        r = self.pos.r  # calculate current radius
        x, y = self.cartesian_pos()  # initial Cartesian position
        vx, vy = self.cartesian_vel()  # initial Cartesian velocity
        delta_t = self.delta_t
        factor = 4 * (np.pi ** 2) / (r ** 3)
        mu = reduced_mass(bodies)

        l = self.ang_mom  # why does this get bigger over time?  # test
        # print('Earth angular momentum at update() method: {}'.format(self.ang_mom))  # test

        # Euler-Cromer step
        vxf = vx - (factor * x * delta_t)
        vyf = vy - (factor * y * delta_t)
        xf = x + (vxf * delta_t)
        yf = y + (vyf * delta_t)

        # update position
        self.pos.r = np.sqrt((xf ** 2) + (yf ** 2))
        self.pos.theta = np.arctan2(yf, xf)

        # update velocity
        self.vel.r = ((xf * vxf) + (yf * vyf)) / self.pos.r  # notice how I update r_dot AFTER updating r
        # self.vel.theta = self.ang_mom() / (mu * (self.pos.r ** 2))  # self.ang_mom() is a constant
        self.vel.theta = l / (mu * (self.pos.r ** 2))

    def run(self):
        # self.calc_ang_mom()  # want to run this method ONCE
        for iteration in range(iterations):
            x, y = self.cartesian_pos()
            self.x_vals.append(x)
            self.y_vals.append(y)
            # self.velocities.append(self.vel)
            self.update()


def main():
    # make the Earth
    earth_pos = Point(1.0, 0)  # in AU
    earth_vel = Vector(0.0, 2.6 * np.pi)  # remember this is in Cartesian coordinates.
    earth_mass = 3.0e-6  # in solar masses
    # earth_l = 1.88e-5  # earth initial angular momentum in solar_masses * AU**2 * rad / yr
    earth = Body(earth_pos, earth_vel, earth_mass, step)

    # make the Sun
    sun_pos = Point(0.0, 0.0)
    # sun_vel = Vector(0.0, 0.0)
    sun_mass = 1.0
    # sun_l = 0.0
    sun = Sun(sun_pos, sun_mass)

    # add all objects to the simulation at once
    earth.add()
    sun.add()

    # calculate angular momentum for Earth
    earth.calc_ang_mom()
    print('Earth angular momentum at initial calculation: {}'.format(earth.ang_mom))

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