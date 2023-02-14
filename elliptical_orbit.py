# import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# time step
iterations = 10000
step = 0.002  # in yr

# specify constants
mass_earth = 3.0e-6
mass_sun = 1.0  # solar masses
mu = mass_earth * mass_sun / (mass_earth + mass_sun)  # reduced mass for convenience
init_rad = 1.0  # in AU
init_theta = 0.0  # in radians
init_r_vel = 0.0  # initial radial velocity of zero
l = 2.5e-5  # constant value. 1.88e-5 results in a roughly circular orbit. solar masses * AU**2 * rad / yr

init_theta_vel = l / (mu * (init_rad ** 2))  # initial theta velocity from l
print('theta_0: {}'.format(init_theta_vel))  # in radians


def cartesian_pos(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def cartesian_vel(r, theta, r_dot, theta_dot):
    # calculate velocities in x and y
    vx = (r_dot * np.cos(theta)) - (r * theta_dot * np.sin(theta))
    vy = (r * theta_dot * np.cos(theta)) + (r_dot * np.sin(theta))

    return vx, vy


x0, y0 = cartesian_pos(init_rad, init_theta)  # initialize
vx0, vy0 = cartesian_vel(init_rad, init_theta, init_r_vel, init_theta_vel)  # initialize


# define function to update positions and velocities using Euler-Cromer method
def euler_cromer(xi, yi, vxi, vyi, delta_t):
    r = np.sqrt(xi ** 2 + yi ** 2)  # calculate current radius
    factor = 4 * (np.pi ** 2) / (r ** 3)
    vxf = vxi - (factor * xi * delta_t)
    vyf = vyi - (factor * yi * delta_t)
    xf = xi + vxf * delta_t
    yf = yi + vyf * delta_t

    return xf, yf, vxf, vyf


# actually update positions and velocities
xdata = []
ydata = []
vxdata = []
vydata = []
times = []

t = 0
for iteration in range(iterations):
    xf, yf, vxf, vyf = euler_cromer(x0, y0, vx0, vy0, step)
    x0, y0, vx0, vy0 = xf, yf, vxf, vyf  # update the values
    xdata.append(xf)
    ydata.append(yf)
    vxdata.append(vxf)  # test
    vydata.append(vyf)  # test
    times.append(t)  # test
    # print('x, y: {}, {}'.format(xf, yf))
    t += step

# test plot
fig, ax = plt.subplots(figsize=(7, 7))
# ax.plot(times, vxdata)
# ax.plot(times, vydata)
ax.plot(times, xdata)
ax.plot(times, ydata)
plt.show()

# now animate
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(7, 7))  # set figure size
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
earth, = ax.plot([], [], 'g.', markersize=15)
sun = ax.plot(0, 0, 'X', markersize=30, color='yellow')
ax.plot(xdata, ydata, 'g--')
plt.grid(True, lw=0.3)  # plot grid
# plt.savefig('output/circular_orbit.png')


def animate(i):
    earth.set_data(xdata[i], ydata[i])
    return earth


animation = FuncAnimation(fig, animate, frames=iterations, interval=200, repeat=True)
# animation.save('output/circular_orbit.gif', writer='pillow')  # mp4 doesn't work
plt.show()
