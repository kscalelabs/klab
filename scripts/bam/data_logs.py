""" Logged data parsing."""
from dataclasses import dataclass
import glob
import numpy as np
import json


@dataclass
class Trajectory:
    trajectory: np.ndarray
    actuator_ids: list[int]
    frequency: float
    amplitude: float
    phase_offset: float
    duration: float
    kp: float
    kd: float
    control_freq: float
    initial_positions: list[float]
    data: list[dict]
    timing_data: dict
    timing_stats: dict
    filename: str

    def get_actuator_positions(self, key="relative_position") -> dict:
        """
        Returns a list of positions for the specified actuator from a single log.
        """
        positions = {"1": [], "2": [], "3": []}
        for timestep in self.data:
            for actuator_idx in self.actuator_ids:
                position = timestep["actuators"][str(actuator_idx)][key]
                positions[str(actuator_idx)].append(position)
        return positions


class Logs:
    def __init__(self, directory: str):
        """ 
        Args:
            directory: The directory to read the logs from.
        """
        self.directory: str = directory
        self.json_files = glob.glob(f"{self.directory}/*.json")

        self.logs = []
        for json_file in self.json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["filename"] = json_file
                self.logs.append(Trajectory(
                    trajectory=np.array(data["data"]),
                    actuator_ids=data["actuator_ids"],
                    frequency=data["frequency"],
                    amplitude=data["amplitude"],
                    phase_offset=data["phase_offset"],
                    duration=data["duration"],
                    kp=data["kp"],
                    kd=data["kd"],
                    control_freq=data["control_freq"],
                    initial_positions=data["initial_positions"],
                    data=data["data"],
                    timing_data=data["timing_data"],
                    timing_stats=data["timing_stats"],
                    filename=json_file
                ))
