from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_FILE = Path(__file__).parent / "config.yaml"
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "benchmark"
EXPERIMENTS_FILE = BENCHMARK_DIR / "experiments.yaml"


@dataclass
class Config:
    """
    Configuration of a model evaluation.

    Parameters
    ----------
    model : str
        Name of the model. It must match the directory `models/<model>`.
    domains : list[str]
        Domains to evaluate.
    experiments : dict[str, list[str] | None]
        Training experiment -> test experiments to evaluate. None (an empty entry
        in the YAML file) means all the tests of that training experiment defined
        in `benchmark/experiments.yaml`. A plain list of training experiments
        (the former format) is also accepted and means all their tests.
    variables : list[str]
        Variables to evaluate.
    max_members : int, default 5
        Maximum number of ensemble members, randomly selected (with a fixed seed)
        by `ml-eval benchmark`, used to score ensemble predictions. 1 scores a
        single random member. Ignored for deterministic models.
    """

    model: str
    domains: list[str] = field(default_factory=lambda: ["ALPS", "NZ", "SA"])
    experiments: dict[str, list[str] | None] = field(
        default_factory=lambda: {
            "ESD_pseudo_reality": [
                "Perfect_cross_validation",
                "Perfect_extrapolation",
            ],
            "Emulator_hist_future": [
                "Perfect_cross_validation",
                "Perfect_interpolation",
            ],
        }
    )
    variables: list[str] = field(default_factory=lambda: ["tasmax", "pr"])
    max_members: int = 5

    def __post_init__(self) -> None:
        """
        Convert `experiments` to a dictionary of lists and check it against
        `benchmark/experiments.yaml`.

        Training or test experiments not defined in that file are reported with a
        warning and removed, so `experiments` only holds valid combinations.

        Raises
        ------
        ValueError
            If `max_members` is not a positive integer.
        """
        # bool is a subclass of int, so exclude it explicitly
        if (
            not isinstance(self.max_members, int)
            or isinstance(self.max_members, bool)
            or self.max_members < 1
        ):
            raise ValueError(
                f"'max_members' must be a positive integer, got {self.max_members!r}"
            )

        if isinstance(self.experiments, list):
            self.experiments = {train_exp: None for train_exp in self.experiments}
        if not isinstance(self.experiments, dict):
            raise TypeError(
                "'experiments' must map each training experiment to a list of "
                f"test experiments, got {type(self.experiments).__name__}"
            )

        with open(EXPERIMENTS_FILE) as f:
            experiment_mapping = yaml.safe_load(f)

        valid = {}
        for train_exp, tests in self.experiments.items():
            if train_exp not in experiment_mapping:
                print(
                    f"[config] WARNING | Training experiment {train_exp!r} not in "
                    f"{EXPERIMENTS_FILE}; skipped"
                )
                continue
            available = experiment_mapping[train_exp]
            if tests is None:
                tests = list(available['tests'])
            elif not isinstance(tests, list):
                raise TypeError(
                    f"The tests of '{train_exp}' must be a list, "
                    f"got {type(tests).__name__}"
                )
            for test_exp in tests:
                if test_exp not in available['tests']:
                    print(
                        f"[config] WARNING | Test {test_exp!r} of {train_exp!r} not in "
                        f"{EXPERIMENTS_FILE}; skipped"
                    )
            valid_tests = [test_exp for test_exp in tests if test_exp in available['tests']]
            if valid_tests:
                valid[train_exp] = valid_tests
        self.experiments = valid

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_FILE) -> Config:
        """
        Read the configuration from a YAML file.

        Attributes that are missing or empty in the file take their default
        values, except for `model`, which is compulsory.

        Parameters
        ----------
        path : str | Path
            YAML configuration file. By default, `config.yaml` next to this module.

        Returns
        -------
        Config
            Configuration read from the file.

        Raises
        ------
        ValueError
            If `model` is not defined in the file.
        TypeError
            If the file contains an unknown attribute.
        """
        with open(path) as f:
            settings = yaml.safe_load(f) or {}

        # a key whose items are all commented out is read by YAML as None
        settings = {key: value for key, value in settings.items() if value is not None}

        if "model" not in settings:
            raise ValueError(f"The attribute 'model' is compulsory in {path}")

        return cls(**settings)
