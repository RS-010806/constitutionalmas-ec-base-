import os
from datetime import datetime
import json


class RunLogger:
    def __init__(self, run_name: str = "run"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = "logs"
        os.makedirs(self.directory, exist_ok=True)
        self.run_id = f"{run_name}_{timestamp}"
        self.log_path = os.path.join(self.directory, f"{self.run_id}.log")
        self.artifact_path = os.path.join(self.directory, f"{self.run_id}.json")

    def log_line(self, text: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def write_artifact(self, data: dict):
        with open(self.artifact_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
