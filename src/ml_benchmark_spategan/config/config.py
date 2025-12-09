import logging
import os
import random
import string
from datetime import datetime

import yaml
from omegaconf import OmegaConf

# Initialize logger
logger = logging.getLogger(__name__)


def generate_run_id(length=8):
    """
    Generate a run ID in format YYYYMMDD_HHMM_<random_string>.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    letters = string.ascii_lowercase + string.digits
    random_suffix = "".join(random.choice(letters) for i in range(length))
    return f"{timestamp}_{random_suffix}"


def setup_logging(config: dict = None):
    """
    Set up logging configuration based on config.
    """
    if config and "logging" in config:
        level = getattr(logging, config.logging.get("level", "INFO").upper())
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config_from_yaml(file_path: str) -> dict:
    """
    Load configuration parameters from a YAML file. Convert to omega config.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    config = OmegaConf.create(config)
    setup_logging(config)
    return config


def save_config_to_yaml(config: dict, file_path: str):
    """
    Save configuration parameters to a YAML file.
    """
    # Convert OmegaConf to regular dict if needed
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)

    with open(file_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    return None


def setup_experiment_directory(base_dir: str, run_id: str) -> str:
    """
    Set up a directory for the experiment.
    """
    experiment_dir = os.path.join(base_dir / "runs", run_id)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def print_config(config: dict):
    """
    Log the configuration in a readable format.
    """
    logger.info("Configuration:\n" + OmegaConf.to_yaml(config))


def set_up_run(project_base: str) -> str:
    """
    Set up the run directory and save the configuration.
    """
    cf = load_config_from_yaml(os.path.join(project_base, "config.yml"))
    print_config(cf)
    run_id = generate_run_id()
    logger.info(f"Run ID: {run_id}")
    # add run_id to the config
    cf.run_id = run_id
    run_dir = setup_experiment_directory(project_base, run_id)
    # add run_dir to the config
    cf.run_dir = run_dir
    # Save the configuration to the run directory
    config_path = os.path.join(run_dir, "config.yaml")
    save_config_to_yaml(cf, config_path)

    logger.info(f"Run directory set up at: {run_dir}")

    return cf
