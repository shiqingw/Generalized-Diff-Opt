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

from cores.utils.config import Configuration
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, solve_LQR_tracking, save_dict, load_dict, init_prosuite_qp
from cores.diff_optimization.diff_opt_helper import DiffOptHelper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=2, type=int, help='test case number')
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
    a = 2*np.pi/(horizon_length*dt)
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

    Q = np.diag([100,100,0,10,10,0])
    Q_list = np.array([Q]*(horizon_length +1))
    R = np.diag([1,1])
    R_list = np.array([R]*(horizon_length))
    print("==> Solve LQR gain")
    K_gains, k_feedforward = solve_LQR_tracking(A_list, B_list, Q_list, R_list, x_traj, horizon_length)

    def lqr_controller(state, i):
        K = K_gains[i]
        k = k_feedforward[i]
        return K @ state + k + u_traj[i,:]

    # Obstacles
    obstacle_config = test_settings["obstacle_config"]
    obstacles = []
    circle = Circle(obstacle_config["ball_position"], obstacle_config["ball_radius"], 
                    facecolor="tab:blue", alpha=1, edgecolor="black", linewidth=1, zorder=1.8)
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

    # Define proxuite problem
    print("==> Define proxuite problem")
    cbf_qp = init_prosuite_qp(n_v=system.n_controls, n_eq=0, n_in=system.n_controls+1)

    # CBF parameters
    CBF_config = test_settings["CBF_config"]
    alpha0 = CBF_config["alpha0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]

    # Create records
    print("==> Create records")
    times = np.linspace(0, horizon_length*dt, horizon_length+1)
    states = np.zeros([horizon_length+1, system.n_states], dtype=config.np_dtype)
    states[0,:] = x_traj[0,:] + np.array([0,1,0,0,0,0])
    controls = np.zeros([horizon_length, system.n_controls], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon_length, system.n_controls], dtype=config.np_dtype)
    phi1s = np.zeros(horizon_length, dtype=config.np_dtype)
    phi2s = np.zeros(horizon_length, dtype=config.np_dtype)
    cbf_values = np.zeros(horizon_length, dtype=config.np_dtype)
    time_cvxpy = np.zeros(horizon_length, dtype=config.np_dtype)
    time_diff_helper = np.zeros(horizon_length, dtype=config.np_dtype)
    time_cbf_qp = np.zeros(horizon_length, dtype=config.np_dtype)
    time_control_loop = np.zeros(horizon_length, dtype=config.np_dtype)

    # Forward simulate the system
    print("==> Forward simulate the system")
    for i in range(horizon_length):
        time_control_loop_start = time.time()
        state = states[i,:]
        time_step = i
        drone_pos_np = state[0:2]
        drone_ori_np = state[2]

        # Pass parameter values to cvxpy problem
        R_b_to_w_np = np.array([[np.cos(drone_ori_np), -np.sin(drone_ori_np)],
                            [np.sin(drone_ori_np), np.cos(drone_ori_np)]])
        ellipse_Q_sqrt_np = ellipse_coef_sqrt_np @ R_b_to_w_np.T
        ellipse_Q_np = ellipse_Q_sqrt_np.T @ ellipse_Q_sqrt_np
        _ellipse_Q_sqrt.value = ellipse_Q_sqrt_np
        _ellipse_b.value = -2 * ellipse_Q_np @ drone_pos_np
        _ellipse_c.value = drone_pos_np.T @ ellipse_Q_np @ drone_pos_np

        time_cvxpy_start = time.time()
        problem.solve(solver=cp.ECOS)
        time_cvxpy_end = time.time()

        # Evaluate gradient and hessian
        dual_val = np.array([problem.constraints[i].dual_value for i in range(len(problem.constraints))]).squeeze()
        alpha_val = _alpha.value
        p_val = np.array(_p.value)
        theta_val = np.array([drone_pos_np[0], drone_pos_np[1], drone_ori_np])

        time_diff_helper_start = time.time()
        grad_alpha_tmp, _, _, hessian_alpha_tmp, _, _ = diff_helper.get_gradient_and_hessian(alpha_val, p_val, theta_val, dual_val)
        time_diff_helper_end = time.time()

        # Evaluate CBF
        CBF = _alpha.value - alpha0

        if CBF_config["active"]:
            # Construct CBF-QP
            grad_alpha = np.zeros(system.n_states, dtype=config.np_dtype)
            grad_alpha[0:len(grad_alpha_tmp)] = grad_alpha_tmp
            hessian_alpha = np.zeros([system.n_states, system.n_states], dtype=config.np_dtype)
            hessian_alpha[0:len(hessian_alpha_tmp), 0:len(hessian_alpha_tmp)] = hessian_alpha_tmp
            drift = system.drift(state) # f(x)
            actuation = system.actuation(state) # g(x)
            drift_jac = system.drift_jac(state) # df/dx
            phi1 = grad_alpha @ drift + gamma1 * CBF

            u_nominal = lqr_controller(state, time_step)
            H = np.eye(system.n_controls)
            g = -u_nominal
            C = np.zeros([system.n_controls+1,system.n_controls], dtype=config.np_dtype)
            C[0,:] = grad_alpha @ drift_jac @ actuation
            C[1:system.n_controls+1,:] = np.eye(system.n_controls)
            lb = np.zeros(system.n_controls+1, dtype=config.np_dtype)
            ub = np.zeros(system.n_controls+1, dtype=config.np_dtype) 
            lb[0] = -drift.T @ hessian_alpha @ drift - grad_alpha @ drift_jac @ drift \
                - (gamma1+gamma2) * grad_alpha @ drift - gamma1 * gamma2 * CBF
            ub[0] = np.inf
            lb[1] = system.u1_constraint[0]
            ub[1] = system.u1_constraint[1]
            lb[2] = system.u2_constraint[0]
            ub[2] = system.u2_constraint[1]
            cbf_qp.update(H=H, g=g, C=C, l=lb, u=ub)
            time_cbf_qp_start = time.time()
            cbf_qp.solve()
            time_cbf_qp_end = time.time()
            u_safe = cbf_qp.results.x
            phi2 = -lb[0] + C[0,:] @ u_safe
        else:
            u_nominal = lqr_controller(state, time_step)
            u_safe = u_nominal
            phi1 = 0
            phi2 = 0
            time_cbf_qp_start = 0
            time_cbf_qp_end = 0

        time_control_loop_end = time.time()

        # Step the system
        new_state = system.get_next_state(state, u_safe)

        # Record
        states[i+1,:] = new_state
        controls[i,:] = u_safe
        desired_controls[i,:] = u_nominal
        cbf_values[i] = CBF
        phi1s[i] = phi1
        phi2s[i] = phi2
        time_cvxpy[i] = time_cvxpy_end - time_cvxpy_start
        time_diff_helper[i] = time_diff_helper_end - time_diff_helper_start
        time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
        time_control_loop[i] = time_control_loop_end - time_control_loop_start

    # Create animation
    print("==> Create animation")
    save_video_path = "{}/video.mp4".format(results_dir)
    system.animate_robot(states, controls, dt, save_video_path, plot_bounding_ellipse=True, plot_traj=True, obstacles=obstacles)

    # Save summary
    print("==> Save results")
    summary = {"times": times,
               "states": states,
               "controls": controls,
               "desired_controls": desired_controls,
               "x_traj": x_traj,
               "u_traj": u_traj,
               "K_gains": K_gains,
               "k_feedforward": k_feedforward,
               "phi1s": phi1s,
               "phi2s": phi2s,
               "cbf_values": cbf_values}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))
    
    # Draw plots
    print("==> Draw plots")
    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, x_traj[:,0], color="tab:blue", linestyle=":", label="x desired")
    plt.plot(times, x_traj[:,1], color="tab:green", linestyle=":", label="y desired")
    plt.plot(times, states[:,0], color="tab:blue", linestyle="-", label="x")
    plt.plot(times, states[:,1], color="tab:green", linestyle="-", label="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_traj_x_y.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({'pdf.fonttype': 42})
    plt.plot(times, states[:,0], linestyle="-", label=r"$x$")
    plt.plot(times, states[:,1], linestyle="-", label=r"$y$")
    plt.plot(times, states[:,2], linestyle="-", label=r"$\theta$")
    plt.plot(times, states[:,3], linestyle="-", label=r"$v_x$")
    plt.plot(times, states[:,4], linestyle="-", label=r"$v_y$")
    plt.plot(times, states[:,5], linestyle="-", label=r"$\omega$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_traj.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(controls)], desired_controls[:,0], color="tab:blue", linestyle=":", 
             label="u_1 nominal")
    plt.plot(times[:len(controls)], desired_controls[:,1], color="tab:green", linestyle=":",
              label="u_2 nominal")
    plt.plot(times[:len(controls)], controls[:,0], color="tab:blue", linestyle="-", label="u_1")
    plt.plot(times[:len(controls)], controls[:,1], color="tab:green", linestyle="-", label="u_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_controls.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(phi1s)], phi1s, label="phi1")
    plt.plot(times[:len(phi2s)], phi2s, label="phi2")
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_phi.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(cbf_values)], cbf_values, label="CBF value")
    plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_cbf.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_cvxpy)], time_cvxpy, label="cvxpy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cvxpy.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_diff_helper)], time_diff_helper, label="diff helper")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_diff_helper.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_cbf_qp)], time_cbf_qp, label="CBF-QP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cbf_qp.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times[:len(time_control_loop)], time_control_loop, label="control loop")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_control_loop.pdf'))
    plt.close(fig)

    # Print solving time
    print("==> Control loop solving time: {:.5f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY solving time: {:.5f} s".format(np.mean(time_cvxpy)))
    print("==> Diff helper solving time: {:.5f} s".format(np.mean(time_diff_helper)))
    print("==> CBF-QP solving time: {:.5f} s".format(np.mean(time_cbf_qp)))



    
