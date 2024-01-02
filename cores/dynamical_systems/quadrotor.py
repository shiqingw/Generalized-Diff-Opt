import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.animation as animation
import IPython


class Quadrotor():
    def __init__(self, prop_dict, params_dict):
        """
        See https://cookierobotics.com/052/ for the derivation of the dynamics
        """
        self.n_states = prop_dict['n_states']
        self.n_controls = prop_dict['n_controls']
        self.mass = params_dict['mass']
        self.inertia = params_dict['inertia']
        self.length = params_dict['length']
        self.gravity = params_dict['gravity']
        self.delta_t = params_dict['delta_t']

        x, y, theta, vx, vy, omega = sp.symbols('x y theta v_x v_y omega')
        u1, u2 = sp.symbols('u_1 u_2')

        drift = sp.Matrix([vx,
                           vy,
                           omega,
                           0,
                           -self.gravity,
                           0])
        self.drift_func = sp.lambdify((x, y, theta, vx, vy, omega), drift, 'numpy')

        drift_jac = drift.jacobian([x, y, theta, vx, vy, omega])
        self.drift_jac_func = sp.lambdify((x, y, theta, vx, vy, omega), drift_jac, 'numpy')

        actuation1 = sp.Matrix([-sp.sin(theta)/ self.mass,
                               sp.cos(theta)/ self.mass,
                               self.length/self.inertia,
                               0,
                               0,
                               0])
    
        actuation2 = sp.Matrix([-sp.sin(theta)/ self.mass,
                               sp.cos(theta)/ self.mass,
                               -self.length/self.inertia,
                               0,
                               0,
                               0])
        self.actuation1_func = sp.lambdify((x, y, theta, vx, vy, omega), actuation1, 'numpy')
        self.actuation2_func = sp.lambdify((x, y, theta, vx, vy, omega), actuation2, 'numpy')

    def drift(self, x):
        """
        x: state vector = [x, y, theta, vx, vy omega]
        """
        return self.drift_func(*x)

    def actuation(self, x):
        """
        x: state vector = [x, y, theta, vx, vy omega]
        """
        g1 = self.actuation1_func(*x)
        g2 = self.actuation2_func(*x)
        return np.hstack((g1, g2))
    
    def drift_jac(self, x, u):
        """
        x: state vector = [x, y, theta, vx, vy omega]
        """
        return self.drift_jac_func(*x)
    
    def get_next_state(self, x, u):
        """
        Inputs:
        x: state of the quadrotor as a numpy array [x, y, theta, vx, vy omega]
        u: control as a numpy array (u1, u2)

        Output:
        the new state of the quadrotor as a numpy array
        """
        dxdt = np.squeeze(self.drift(x)) + self.actuation(x) @ u
        x_next = x + self.delta_t*dxdt
        return x_next
    
    def simulate(self, x0, controller, horizon_length, disturbance=False):
        """
        This function simulates the quadrotor for horizon_length steps from initial state z0

        Inputs:
        x0: the initial conditions of the quadrotor as a numpy array [x, y, theta, vx, vy omega]
        controller: a function that takes a state z as argument and index i of the time step and returns a control u
        horizon_length: the horizon length

        disturbance: if True will generate a random push every seconds during the simulation

        Output:
        t[time_horizon+1] contains the simulation time
        x[time_horizon+1, n_states, ] and u[time_horizon, n_controls] containing the time evolution of states and control
        """
    
        t = np.zeros([horizon_length+1,])
        x = np.empty([horizon_length+1, self.n_states])
        x[:,0] = x0
        u = np.zeros([horizon_length, self.n_controls])
        for i in range(horizon_length):
            u[i,:] = controller(x[i,:],i)
            x[i+1,:] = self.get_next_state(x[i,:], u[i,:])
            if disturbance and np.mod(i,100)==0:
                dist = np.zeros([self.n_states, ])
                dist[1::2] = np.random.uniform(-1.,1,(3,))
                x[i+1,:] += dist
            t[i+1] = t[i] + self.delta_t
        return t, x, u
    
    def animate_robot(self, x, u, dt, save_video_path):
        """
        This function makes an animation showing the behavior of the quadrotor
        takes as input the result of a simulation (with dt=0.01s)
        """

        min_dt = 0.1
        if(dt < min_dt):
            steps = int(min_dt/dt)
            use_dt = int(np.round(min_dt * 1000)) #in ms
        else:
            steps = 1
            use_dt = int(np.round(dt * 1000)) #in ms

        #what we need to plot
        plotx = x[::steps,:]
        plotu = u[::steps,:]
        plotx = plotx[:len(plotu)]

        fig, ax = plt.subplots(figsize=[8.5, 8.5])
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.grid()

        list_of_lines = []

        #create the robot
        # the main frame
        line, = ax.plot([], [], 'k', lw=6)
        list_of_lines.append(line)
        # the left propeller
        line, = ax.plot([], [], 'b', lw=4)
        list_of_lines.append(line)
        # the right propeller
        line, = ax.plot([], [], 'b', lw=4)
        list_of_lines.append(line)
        # the left thrust
        line, = ax.plot([], [], 'r', lw=1)
        list_of_lines.append(line)
        # the right thrust
        line, = ax.plot([], [], 'r', lw=1)
        list_of_lines.append(line)

        def _animate(i):
            for l in list_of_lines: #reset all lines
                l.set_data([],[])

            x = plotx[i,0]
            y = plotx[i,1]
            theta = plotx[i,2]
            trans = np.array([[x,x],[y,y]])
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            main_frame = np.array([[-self.length, self.length],
                                   [0,0]])
            main_frame = rot @ main_frame + trans 

            left_propeller = np.array([[-1.3 * self.length, -0.7*self.length],
                                       [0.1,0.1]])
            left_propeller = rot @ left_propeller + trans

            right_propeller = np.array([[1.3 * self.length, 0.7*self.length],
                                        [0.1,0.1]])
            right_propeller = rot @ right_propeller + trans

            right_thrust = np.array([[self.length, self.length],
                                     [0.1, 0.1+plotu[i,0]*0.04]])
            right_thrust = rot @ right_thrust + trans

            left_thrust = np.array([[-self.length, -self.length],
                                    [0.1, 0.1+plotu[i,1]*0.04]])
            left_thrust = rot @ left_thrust + trans

            list_of_lines[0].set_data(main_frame[0,:], main_frame[1,:])
            list_of_lines[1].set_data(left_propeller[0,:], left_propeller[1,:])
            list_of_lines[2].set_data(right_propeller[0,:], right_propeller[1,:])
            list_of_lines[3].set_data(left_thrust[0,:], left_thrust[1,:])
            list_of_lines[4].set_data(right_thrust[0,:], right_thrust[1,:])

            return list_of_lines

        def _init():
            return _animate(0)

        ani = animation.FuncAnimation(fig, _animate, np.arange(0, len(plotx)),
            interval=use_dt, blit=True, init_func=_init)
        plt.close(fig)
        plt.close(ani._fig)
        ani.save(save_video_path, writer='ffmpeg', fps=int(1000/use_dt))
