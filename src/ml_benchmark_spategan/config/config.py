import yaml
import os
import random
import string
from omegaconf import OmegaConf

def generate_run_id(length=8):
    """
    Generate a random string of fixed length.
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

def load_config_from_yaml(file_path: str) -> dict:
    """
    Load configuration parameters from a YAML file. Convert to omega config.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config = OmegaConf.create(config)
    return config

def save_config_to_yaml(config: dict, file_path: str):
    """
    Save configuration parameters to a YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.dump(config, file)
    return None

def setup_experiment_directory(base_dir: str, run_id: str) -> str:
    """
    Set up a directory for the experiment.
    """
    experiment_dir = os.path.join(base_dir, run_id)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir