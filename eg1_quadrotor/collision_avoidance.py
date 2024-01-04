import json
import sys
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.patches import Circle
import time
import sympy as sp
import proxsuite

from cores.utils.config import Configuration
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, solve_LQR_tracking, save_dict, load_dict
from cores.diff_optimization.diff_opt_helper import DiffOptHelper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/results_eg1/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    # Load test settings
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Seed everything
    seed_everything(test_settings["seed"])

    # Load configuration
    config = Configuration()

    # Build dynamical system
    system_name = test_settings["system_name"]
    system = get_system(system_name)

    # Tracking control via LQR
    horizon = 10
    dt = system.delta_t
    horizon_length = int(horizon/dt)

    x_traj = np.zeros([horizon_length+1, system.n_states, ])
    t = np.linspace(0,horizon_length*dt, horizon_length+1)
    a = 3*np.pi/(horizon_length*dt)
    semi_major_axis = 2
    semi_minor_axis = 1
    x_traj[:,0] = semi_major_axis * np.cos(a*t) # x
    x_traj[:,1] = semi_minor_axis * np.sin(a*t) # y
    x_traj[:,3] = -semi_major_axis * a * np.sin(a*t) # xdot
    x_traj[:,4] = semi_minor_axis * a * np.cos(a*t) # ydot
    u_traj = 0.5 * system.mass * system.gravity * np.ones([horizon_length, system.n_controls])
    A_list = np.empty((horizon_length, system.n_states, system.n_states))
    B_list = np.empty((horizon_length, system.n_states, system.n_controls))
    for i in range(horizon_length):
        A, B = system.get_linearization(x_traj[i,:], u_traj[i,:]) 
        A_list[i] = A
        B_list[i] = B

    Q = np.diag([10,10,0,10,10,0])
    Q_list = np.array([Q]*(horizon_length +1))
    R = np.diag([1,1])
    R_list = np.array([R]*(horizon_length))
    print("==> Solve LQR gain")
    K_gains, k_feedforward = solve_LQR_tracking(A_list, B_list, Q_list, R_list, x_traj, horizon_length)

    # Obstacles
    obstacle_config = test_settings["obstacle_config"]
    obstacles = []
    circle = Circle(obstacle_config["ball_position"], obstacle_config["ball_radius"], 
                    facecolor="tab:blue", alpha=1, edgecolor="black", linewidth=1, zorder=2)
    obstacles.append(circle)

    # Bounding shapes
    ellipse_coef_sqrt_np = np.array([1.0/system.bounding_shape_config["semi_major_axis"], 
                                    1.0/system.bounding_shape_config["semi_minor_axis"]], dtype=config.np_dtype)
    ellipse_coef_sqrt_np = np.diag(ellipse_coef_sqrt_np)
    
    # Define cvxpy problem
    print("==> Define cvxpy problem")
    _p = cp.Variable(2)
    _alpha = cp.Variable(1, pos=True)
    _ellipse_Q_sqrt = cp.Parameter((2,2))
    _ellipse_b = cp.Parameter(2)
    _ellipse_c = cp.Parameter()
    ball_center_np = np.array(obstacle_config["ball_position"])
    ball_radius_np = obstacle_config["ball_radius"]
    obj = cp.Minimize(_alpha)
    cons = [cp.sum_squares(_p - ball_center_np)/ball_radius_np**2 <= _alpha,
            cp.sum_squares(_ellipse_Q_sqrt @ _p) + _ellipse_b.T @ _p + _ellipse_c <= _alpha]
    problem = cp.Problem(obj, cons)
    assert problem.is_dcp()
    assert problem.is_dpp()

    # Define diff helper
    print("==> Define diff helper")
    px, py, alpha, cx, cy, theta = sp.symbols('px py alpha cx cy theta', real=True)
    p_vars = [px, py]
    theta_vars = [cx, cy, theta]
    p = sp.Matrix(p_vars)
    ball_center_tmp = ball_center_np[:, np.newaxis]
    con1 = (p - ball_center_tmp).T @ (p - ball_center_tmp)/ball_radius_np**2
    R_b_to_w = sp.Matrix([[sp.cos(theta), -sp.sin(theta)],
                            [sp.sin(theta), sp.cos(theta)]])
    con2 = (p-sp.Matrix([cx, cy])).T @ R_b_to_w @ ellipse_coef_sqrt_np.T @ ellipse_coef_sqrt_np @ R_b_to_w.T @ (p-sp.Matrix([cx, cy])) 
    cons = [sp.simplify(con1)[0,0], sp.simplify(con2)[0,0]]
    diff_helper = DiffOptHelper(cons, p_vars, theta_vars)

    # Pass parameter values to cvxpy problem
    drone_ori_np = 0.0
    drone_pos_np = np.array([2.0,1.0], dtype=config.np_dtype)
    R_b_to_w_np = np.array([[np.cos(drone_ori_np), -np.sin(drone_ori_np)],
                         [np.sin(drone_ori_np), np.cos(drone_ori_np)]])
    ellipse_Q_sqrt_np = ellipse_coef_sqrt_np @ R_b_to_w_np.T
    ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
    _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
    _ellipse_b.value = -2 * ellipse_Q_np @ drone_pos_np
    _ellipse_c.value = drone_pos_np.T @ ellipse_Q_np @ drone_pos_np

    time1 = time.time()
    problem.solve(solver=cp.SCS)
    time2 = time.time()
    print("Time for SCS solving: ", time2 - time1)
    print(_alpha.value, _p.value)

    # Evaluate gradient and hessian
    dual_val = np.array([problem.constraints[i].dual_value for i in range(len(problem.constraints))]).squeeze()
    alpha_val = _alpha.value
    p_val = np.array(_p.value)
    theta_val = np.array([drone_pos_np[0], drone_pos_np[1], drone_ori_np])

    time1 = time.time()
    grad_alpha, _, _, hessian_alpha, _, _ = diff_helper.get_gradient_and_hessian(alpha_val, p_val, theta_val, dual_val)
    time2 = time.time()
    print("Time for getting gradient and hessian: ", time2 - time1)
    print("Gradient of alpha: ", grad_alpha)
    print("Hessian of alpha: ", hessian_alpha)

    # Construct CBF
    print("==> Construct CBF")

    # def lqr_controller(state, i):
    #     K = K_gains[i]
    #     k = k_feedforward[i]
    #     return K @ state + k + u_traj[i,:]

    # print("==> Forward simulate the system")
    # x0 = x_traj[0]
    # t, x, u = system.simulate(x0, lqr_controller, horizon_length, disturbance=False)

    # print("==> Create video")
    # save_video_path = "{}/video.mp4".format(results_dir)
    # system.animate_robot(x, u, dt, save_video_path, plot_bounding_ellipse=True, plot_traj=True, obstacles=obstacles)

    # print("==> Save results")
    # summary = {"times": t,
    #            "states": x,
    #            "controls": u,
    #            "x_traj": x_traj,
    #            "u_traj": u_traj,
    #            "K_gains": K_gains,
    #            "k_feedforward": k_feedforward}
    # save_dict(summary, os.path.join(results_dir, 'summary.pkl'))
    
    # print("==> Draw plots")
    # fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    # plt.plot(t, x_traj[:,0], color="tab:blue", linestyle=":", label="x desired")
    # plt.plot(t, x_traj[:,1], color="tab:green", linestyle=":", label="y desired")
    # plt.plot(t, x[:,0], color="tab:blue", linestyle="-", label="x")
    # plt.plot(t, x[:,1], color="tab:green", linestyle="-", label="y")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, 'plot_traj_x_y.pdf'))
    # plt.close(fig)

    # fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    # plt.plot(t[:len(u)], u[:,0], label="u_1")
    # plt.plot(t[:len(u)], u[:,1], label="u_2")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(results_dir, 'plot_controls.pdf'))
    # plt.close(fig)



    
