import json
import sys
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from cores.utils.config import Configuration
from cores.dynamical_systems.create_system import get_system
from cores.utils.utils import seed_everything

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

    config = Configuration()

    # Build dynamical system
    system_name = test_settings["system_name"]
    system = get_system(system_name)

    horizon = 10
    dt = 0.01
    states = np.zeros((int(horizon/dt)+1, system.n_states))
    controls = np.zeros((int(horizon/dt), system.n_controls))
    controls[:,0]+=10
    controls[:,1]+=5
    save_video_path = "{}/video.mp4".format(results_dir)
    system.animate_robot(states, controls, dt, save_video_path)