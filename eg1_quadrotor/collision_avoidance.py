import json
import sys
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt

from cores.utils.config import Configuration
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything, solve_LQR_trajectory, save_dict, load_dict

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
    seed_everything(0)
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    config = Configuration()

    # Build dynamical system
    system_name = test_settings["system_name"]
    system = get_system(system_name)

    # Tracking trajectory
    horizon = 10
    dt = system.delta_t
    horizon_length = int(horizon/dt)

    x_traj = np.zeros([horizon_length+1, system.n_states, ])
    t = np.linspace(0,horizon_length*dt, horizon_length+1)
    a = 3*np.pi/(horizon_length*dt)
    radius = 2
    x_traj[:,0] = radius * np.cos(a*t) 
    x_traj[:,1] = radius * np.sin(a*t) 
    x_traj[:,3] = -radius * a * np.sin(a*t) 
    x_traj[:,4] = radius * a * np.cos(a*t) 
    u_traj = 0.5 * system.mass * system.gravity * np.ones([horizon_length, system.n_controls])
    A_list = []
    B_list = []
    for i in range(horizon_length):
        A, B = system.get_linearization(x_traj[i,:], u_traj[i,:]) 
        A_list.append(A)
        B_list.append(B)

    Q = np.diag([10,10,0,10,10,0])
    Q_list = [Q]*(horizon_length +1)
    R = np.diag([1,1])
    R_list = [R]*horizon_length
    print("==> Solve LQR gain")
    K_gains, k_feedforward = solve_LQR_trajectory(A_list, B_list, Q_list, R_list, x_traj, horizon_length)

    def lqr_controller(state, i):
        K = K_gains[i]
        k = k_feedforward[i]
        return K @ state + k + u_traj[i,:]

    print("==> Forward simulate the system")
    x0 = np.zeros(system.n_states)
    t, x, u = system.simulate(x0, lqr_controller, horizon_length, disturbance=False)

    print("==> Create video")
    save_video_path = "{}/video.mp4".format(results_dir)
    system.animate_robot(x, u, dt, save_video_path)

    print("==> Save results")
    summary = {"times": t,
               "states": x,
               "controls": u,
               "x_traj": x_traj,
               "u_traj": u_traj,
               "K_gains": K_gains,
               "k_feedforward": k_feedforward}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))
    
    print("==> Draw plots")
    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(t, x_traj[:,0], color="tab:blue", linestyle=":", label="x desired")
    plt.plot(t, x_traj[:,1], color="tab:green", linestyle=":", label="y desired")
    plt.plot(t, x[:,0], color="tab:blue", linestyle="-", label="x")
    plt.plot(t, x[:,1], color="tab:green", linestyle="-", label="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_traj_x_y.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(t[:len(u)], u[:,0], label="u_1")
    plt.plot(t[:len(u)], u[:,1], label="u_2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_controls.pdf'))
    plt.close(fig)



    
