"""
Experiment Tracker Utility
--------------------------
Standardizes the logging of statistical experiments to maintain a 
scientific record of all hypotheses tested.
"""
import json
import os
from datetime import datetime
from typing import Dict, Any

LOG_FILE = "research/experiment_log.json"

def log_experiment(name: str, metrics: Dict[str, Any]):
    """
    Logs an experiment result to the central JSON log.
    """
    if not os.path.exists("research"):
        os.makedirs("research")

    entry = {
        "experiment": name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    }

    data = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    data.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Experiment '{name}' logged successfully.")
