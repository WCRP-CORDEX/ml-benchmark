#!/usr/bin/env python
"""
Check that the evaluation of a model is complete.

For every train/test experiment pair in the model's configuration file, check that
`models/<MODEL>/evaluation/<TRAIN_EXP>/<TEST_EXP>/metrics_summary.json` (written by
`evaluate.py`) exists and covers every variable and domain of the configuration.

Problems are reported per experiment:

- ERROR: missing summary, variable or domain. The exit code is then 1.
- WARNING: null values (NaN in the per-domain metrics, which make the aggregate
  null), and a `model` field in the summary that differs from the configuration.
  They are reported, never filled, and do not change the exit code.

Usage
-----
    python evaluation/validate.py models/<MODEL>               # model directory
    python evaluation/validate.py models/<MODEL>/config.yaml   # config file
    python evaluation/validate.py models/*                     # every model
    python evaluation/validate.py                              # evaluation/config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config import DEFAULT_CONFIG_FILE, Config
from evaluate import SUMMARY_FILE, experiment_dir

CONFIG_FILE = "config.yaml"  # name of the config file inside a model directory


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Arguments, with the attribute `paths`.
    """
    parser = argparse.ArgumentParser(
        description="Check that the evaluation of a model is complete."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="model directories (holding config.yaml) or configuration files "
        f"(default: {DEFAULT_CONFIG_FILE})",
    )
    return parser.parse_args()


def config_files(paths: list[Path]) -> list[Path]:
    """
    Configuration files to check, from the command-line paths.

    A model directory stands for its `config.yaml`. Directories without one (e.g.
    a stray folder matched by `models/*`) are skipped with a message. Without
    paths, `DEFAULT_CONFIG_FILE` is checked, as in `evaluate.py`.
    """
    if not paths:
        return [DEFAULT_CONFIG_FILE]
    files = []
    for path in paths:
        if path.is_dir():
            if (path / CONFIG_FILE).is_file():
                files.append(path / CONFIG_FILE)
            else:
                print(f"[validate] INFO | No {CONFIG_FILE} in {path}; skipped")
        else:
            files.append(path)
    return files


def check_summary(summary: dict, config: Config) -> tuple[list[str], list[str]]:
    """
    Compare one metric summary with the variables and domains of a configuration.

    Parameters
    ----------
    summary : dict
        Content of a `metrics_summary.json` file.
    config : Config
        Configuration of the model.

    Returns
    -------
    errors : list[str]
        Missing variables, and domains missing from the metrics of a variable.
    warnings : list[str]
        Null (NaN) domain values, and a `model` field that differs from
        `config.model`.
    """
    errors, warnings = [], []
    if summary.get("model") != config.model:
        warnings.append(
            f"'model' is {summary.get('model')!r}, not {config.model!r}"
        )

    variables = summary.get("variables", {})
    for var in config.variables:
        if var not in variables:
            errors.append(f"variable {var!r} missing")
            continue
        metrics = variables[var]
        if not metrics:
            errors.append(f"variable {var!r} has no metrics")
            continue
        for domain in config.domains:
            # metrics that lack this domain, or whose value for it is null
            missing = [m for m, v in metrics.items() if domain not in v["domains"]]
            null = [
                m for m, v in metrics.items()
                if domain in v["domains"] and v["domains"][domain] is None
            ]
            if len(missing) == len(metrics):
                errors.append(f"{var}: domain {domain!r} missing")
            elif missing:
                errors.append(
                    f"{var}: domain {domain!r} missing in {', '.join(missing)}"
                )
            if null:
                warnings.append(f"{var}: null values in {domain} for {', '.join(null)}")
    return errors, warnings


def validate(config_file: Path) -> int:
    """
    Check every experiment of one model's configuration and print a report.

    Parameters
    ----------
    config_file : Path
        Configuration file of the model.

    Returns
    -------
    int
        Number of errors (missing summaries, variables or domains).
    """
    config = Config.from_yaml(config_file)
    print(f"[validate] INFO | Model {config.model} ({config_file})")
    n_errors = n_checked = 0
    for train_exp, test_exps in config.experiments.items():
        for test_exp in test_exps:
            n_checked += 1
            exp = f"{train_exp} / {test_exp}"
            file = experiment_dir(config, train_exp, test_exp) / SUMMARY_FILE
            if not file.is_file():
                print(f"[validate] ERROR | {exp}: missing {file}")
                n_errors += 1
                continue
            with open(file) as f:
                errors, warnings = check_summary(json.load(f), config)
            for msg in errors:
                print(f"[validate] ERROR | {exp}: {msg}")
            for msg in warnings:
                print(f"[validate] WARNING | {exp}: {msg}")
            n_errors += len(errors)

    status = "complete" if n_errors == 0 else f"{n_errors} error(s)"
    print(
        f"[validate] INFO | {config.model}: {n_checked} experiment(s) checked, {status}"
    )
    return n_errors


def main() -> None:
    """Check the evaluation of every model given on the command line."""
    args = parse_args()
    failed = []
    for config_file in config_files(args.paths):
        if not config_file.is_file():
            print(f"[validate] ERROR | Configuration file not found: {config_file}")
            failed.append(str(config_file))
            continue
        if validate(config_file) > 0:
            failed.append(str(config_file))
    if failed:
        sys.exit(f"Incomplete evaluation: {', '.join(failed)}")


if __name__ == "__main__":
    main()
