import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.animation as animation
import IPython


class Quadrotor():
    def __init__(self, prop_dict, params_dict):
        self.n_states = prop_dict['n_states']
        self.n_controls = prop_dict['n_controls']
        self.mass = params_dict['mass']
        self.inertia = params_dict['inertia']
        self.length = params_dict['length']
        self.gravity = params_dict['gravity']
        self.delta_t = params_dict['delta_t']

        x, vx, y, vy, theta, omega = sp.symbols('x v_x y v_y theta omega')
        u1, u2 = sp.symbols('u_1 u_2')

        drift = sp.Matrix([vx,
                           0,
                           vy,
                           -self.gravity,
                           omega,
                           0])
        drift_func = sp.lambdify((x, vx, y, vy, theta, omega, u1, u2), drift, 'numpy')
        self.drift_func = drift_func

        actuation1 = sp.Matrix([0,
                               -sp.sin(theta)/ self.mass,
                               0,
                               sp.cos(theta)/ self.mass,
                               0,
                               self.length/self.inertia])
    
        actuation2 = sp.Matrix([0,
                               -sp.sin(theta)/ self.mass,
                               0,
                               sp.cos(theta)/ self.mass,
                               0,
                               -self.length/self.inertia])
        actuation_func1 = sp.lambdify((x, vx, y, vy, theta, omega, u1, u2), actuation1, 'numpy')
        actuation_func2 = sp.lambdify((x, vx, y, vy, theta, omega, u1, u2), actuation2, 'numpy')
        self.actuation_func1 = actuation_func1
        self.actuation_func2 = actuation_func2

    def drift(self, x, u):
        """
        x: state vector = [x, vx, y, vy, theta, omega]
        u: control vector = [u1, u2]
        """
        return self.drift_func(*x, *u)

    def actuation(self, x, u):
        """
        x: state vector = [x, vx, y, vy, theta, omega]
        u: control vector = [u1, u2]
        """
        g1 = self.actuation_func1(*x, *u)
        g2 = self.actuation_func2(*x, *u)
        return np.hstack((g1, g2))
