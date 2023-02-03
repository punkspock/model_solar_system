# simple circular orbit using Euler-Cromer method

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass

# constants
step = 600  # seconds
sec_day = 1440 * 60  # seconds per day
iterations = sec_day / step
earth_to_sun = 91.607  # AU
pi = np.pi  # for simplicity

# helpers
@dataclass
class Point:
    x: float
    y: float

class Vector(Point):
    pass


class Planet:
    def __init__(self, pos: Point, vel: Vector):
        self.pos = pos
        self.vel = vel


if __name__ == '__main__':
    earth_init_pos = Point(100, 200)
    x_dot_i = 1000.0  # km/s
    y_dot_i = 1000.0  # km/s
    theta_dot_i = (x_dot_i**2 + y_dot_i**2)**(1/2)  # questioning this


