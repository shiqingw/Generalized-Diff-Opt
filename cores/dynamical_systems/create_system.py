import json
import sys
import os
from .quadrotor import Quadrotor
from pathlib import Path

def get_system(system_name):
    with open(os.path.join(str(Path(__file__).parent), "system_params.json"), 'r') as f:
        data = json.load(f)
    if system_name not in data:
        raise ValueError("System name not found in system_params.json")
    data = data[system_name]
    if data["type"] == "Quadrotor":
        return Quadrotor(data["properties"], data["params"])
    else:
        raise ValueError("System type not found in systems.json")


