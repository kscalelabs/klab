""" Logs parsing."""
import glob
import copy
import json
from dataclasses import dataclass


@dataclass
class CommandStep:
    filename: str
    data: dict


class Logs:
    def __init__(self, directory: str):
        """
        Initializes the Logs object by reading all .json files in the specified directory.
        Each log is appended to self.logs, and we also store the filename under data["filename"].
        """
        self.directory: str = directory
        self.json_files = glob.glob(f"{self.directory}/*.json")

        self.logs = []
        for json_file in self.json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["filename"] = json_file
                self.logs.append(data)

    def split(self, selector_kp: float) -> "Logs":
        """
        Creates a new Logs2 object containing only the logs whose 'kp' field
        matches the given selector_kp. Those logs are removed from this object.
        
        Example:
            logs2 = Logs2("logs2_dir")
            new_logs = logs2.split(selector_kp=60.0)
            
            # now 'new_logs.logs' contains only logs that had kp == 60.0
            # while 'logs2.logs' contains all the other logs
        """
        indices = []
        for i, log in enumerate(self.logs):
            # Safely retrieve 'kp', defaulting to None if missing
            if log.get("kp", None) == selector_kp:
                indices.append(i)

        # Make a deep copy to create a separate Logs2 object
        selected_logs = copy.deepcopy(self)

        # Filter out logs that match selector_kp in the original object
        self.logs = [self.logs[i] for i in range(len(self.logs)) if i not in indices]
        self.json_files = [
            self.json_files[i] for i in range(len(self.json_files)) if i not in indices
        ]

        # Keep only the logs that match selector_kp in the new Logs2 object
        selected_logs.logs = [
            selected_logs.logs[i] for i in range(len(selected_logs.logs)) if i in indices
        ]
        selected_logs.json_files = [
            selected_logs.json_files[i] 
            for i in range(len(selected_logs.json_files)) 
            if i in indices
        ]

        return selected_logs


if __name__ == "__main__":
    logs = Logs2("scripts/bam/data_wesley")
    breakpoint()
    one_run = logs.logs[0]
    breakpoint()