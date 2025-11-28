import os
import json
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt


def plot_history(history_path: str, save_path: Optional[str] = None):
    if not(os.path.exists):
        raise FileNotFoundError(f'No folder exists at the location specified {history_path}')
    else:
        with open(history_path, "r") as fs:
            history = json.load(fs)
    
    nb_iter = len(history["shift"]["mean_reward"])
    mean_reward_shift = np.array(history["shift"]["mean_reward"])
    std_reward_shift = np.array(history["shift"]["std_reward"])
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(nb_iter), np.repeat(history["mean_reward"], nb_iter), "green", label="mean reward original")
    plt.plot(np.arange(nb_iter), mean_reward_shift, "orange", label="mean_reward_shift")
    plt.fill_between(np.arange(nb_iter), mean_reward_shift - std_reward_shift, mean_reward_shift + std_reward_shift, label="std_reward_shift", alpha=0.2, facecolor="orange")
    plt.plot(np.arange(nb_iter), history["performance_drop"], "black", label="performance_drop")
    plt.grid()
    plt.legend()
    plt.show()
    
    if save_path is not None:
        plt.savefig(save_path)


class NpEncoder(json.JSONEncoder):
    """
    taken from : https://java2blog.com/object-of-type-int64-is-not-json-serializable/
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # if the object is a function, save it as a string
        if callable(obj):
            return str(obj)
        return super(NpEncoder, self).default(obj)
    
    
    
    if __name__ == "__main__":
        from domain_shift_kpis import here
        file_name = "history.json"
        file_path = os.path.join(here, "..", "trained_models", "PPO_SB3_FINETUNE")
        plot_history(os.path.join(file_path, file_name))
        