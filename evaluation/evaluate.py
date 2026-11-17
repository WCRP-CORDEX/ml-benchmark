#!/usr/bin/env python
"""
Compute the benchmark metrics of a model with `ml-eval benchmark` and aggregate
them across domains, one train/test experiment at a time.

For every train/test experiment pair, `ml-eval benchmark` writes one metric file
per variable and domain (`metrics_{var}_{domain}.json`), which are then
aggregated across domains into `metrics_summary.json`. Experiments that already
have a summary are skipped unless `--overwrite` is given.

Usage
-----
    ./evaluate.py --config-file config.yaml
    ./evaluate.py -c config.yaml
    ./evaluate.py -c config.yaml --overwrite
    ./evaluate.py -c config.yaml --aggregate-only
    ./evaluate.py -c config.yaml --keep-domain-files
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from config import DEFAULT_CONFIG_FILE, Config

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"  # benchmark data (Bench-data)
MODELS_DIR = ROOT_DIR / "models"  # one directory per model
BENCHMARK_DIR = ROOT_DIR / "benchmark"
SUMMARY_FILE = "metrics_summary.json"


def quadratic_mean(values: np.ndarray) -> float:
    """
    Compute the quadratic mean (root mean square) of an array.

    Parameters
    ----------
    values : np.ndarray
        Values to aggregate.

    Returns
    -------
    float
        sqrt(mean(values**2)); NaN if any value is NaN.
    """
    return float(np.sqrt(np.mean(np.square(values))))


# metric -> aggregation across domains
AGGREGATION = {
    "bias_annstd": np.mean,
    "bias_max1day": np.mean,
    "bias_mean": np.mean,
    "bias_sdii": np.mean,
    "lhd": np.mean, # quadratic mean
    "pss": np.mean,
    "ralsd": np.mean,
    "rmse": np.mean, # quadratic mean
}

# horizon -> 20-year period, as in the benchmark file names
with open(BENCHMARK_DIR / "horizons.yaml") as f:
    PERIODS = yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Arguments, with the attributes `config_file`, `overwrite`,
        `aggregate_only` and `keep_domain_files`.
    """
    parser = argparse.ArgumentParser(
        description="Compute the benchmark metrics of a model."
    )
    parser.add_argument(
        "-c",
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="YAML configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help=f"recompute experiments that have already been evaluated "
        "(by default they are skipped)",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="skip `ml-eval` and only rebuild the summaries from the domain files",
    )
    parser.add_argument(
        "--keep-domain-files",
        action="store_true",
        help="keep the per-domain metric files after writing the summary "
        "(by default they are deleted)",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    """Read a YAML file into a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def experiment_dir(config: Config, train_exp: str, test_exp: str) -> Path:
    """Output directory of one train/test experiment pair."""
    return MODELS_DIR / config.model / "evaluation" / train_exp / test_exp


def metrics_file(out_dir: Path, var: str, domain: str) -> Path:
    """Per-domain metric file written by `ml-eval benchmark`."""
    return out_dir / f"metrics_{var}_{domain}.json"


def input_exists(path: Path) -> bool:
    """
    Whether an input file exists, or a glob pattern matches at least one file.

    `Path.is_file()` treats wildcards (`*`, `?`, `[`) as literal characters, so
    patterns (e.g. `*_century` for the mid- and end-century tests together) are
    resolved with `glob` instead.
    """
    if any(char in str(path) for char in "*?["):
        return bool(glob.glob(str(path)))
    return path.is_file()


def _check_attributes(values: list, name: str, out_dir: Path) -> object:
    """
    Value shared by all the per-domain metric files of an experiment.

    Values may be lists (e.g. the positions of the selected ensemble members);
    they are compared as tuples, since lists cannot be dictionary keys.

    Returns None (null in the JSON), with a warning, if the files disagree; a
    value is never picked silently.
    """
    hashable = [tuple(v) if isinstance(v, list) else v for v in values]
    unique = list(dict.fromkeys(hashable))
    if len(unique) == 1:
        value = unique[0]
        return list(value) if isinstance(value, tuple) else value
    if unique:
        print(f"[aggregate] WARNING | Inconsistent {name!r} {unique} in {out_dir}")
    return None


def _to_json_float(x: float) -> float | None:
    """Convert a scalar to a JSON-safe float (NaN -> None, written as null)."""
    x = float(x)
    return None if np.isnan(x) else x


def evaluate_experiment(
    config: Config,
    train_exp: str,
    test_exp: str,
    attrs: dict,
    dataset_mapping: dict,
    overwrite: bool = False,
) -> int:
    """
    Run `ml-eval benchmark` for every domain and variable of one experiment pair.

    Domains whose reference or prediction file is missing are skipped, and a
    failed `ml-eval` call does not stop the remaining cases.

    Parameters
    ----------
    config : Config
        Model, domains and variables to evaluate, and the maximum number of
        ensemble members (`max_members`) passed to `ml-eval benchmark`.
    train_exp : str
        Training experiment, e.g. `ESD_pseudo_reality`.
    test_exp : str
        Test experiment, e.g. `Perfect_cross_validation`.
    attrs : dict
        Test attributes from `experiments.yaml` (`horizon`, `case`, `transfer`).
    dataset_mapping : dict
        Domain attributes from `datasets.yaml`. The GCM is `test_gcm` if
        `transfer` is true, otherwise `train_gcm`.
    overwrite : bool, default False
        Recompute per-domain metric files that already exist. If False, they are
        reused.

    Returns
    -------
    int
        Number of failed `ml-eval` calls.
    """
    horizon, case = attrs["horizon"], attrs["case"]
    period = PERIODS[horizon]
    horizon = '*_century' if horizon == 'mid_end_century' else horizon
    gcm_key = "test_gcm" if attrs["transfer"] else "train_gcm"
    out_dir = experiment_dir(config, train_exp, test_exp)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_failed = 0
    for domain in config.domains:
        forcing = dataset_mapping[domain][gcm_key]

        # x0: reference file(s)
        x0 = (
            DATA_DIR / "Bench-data" / f"{domain}_domain" / "test" / horizon
            / "target" / f"pr_tasmax_{forcing}_{period}.nc"
        )
        if not input_exists(x0):
            print(f"[evaluate] WARNING | Reference file not found: {x0}")
            continue
        # x1: prediction file(s)
        x1 = (
            MODELS_DIR / config.model / "predictions" / f"{domain}_Domain"
            / train_exp / horizon / case / f"Predictions_pr_tasmax_{forcing}_{period}.nc"
        )
        if not input_exists(x1):
            print(f"[evaluate] WARNING | Prediction file not found: {x1}")
            continue

        # compute benchmark metrics for each variable
        for var in config.variables:
            out_file = metrics_file(out_dir, var, domain)
            if out_file.is_file() and not overwrite:
                print(f"[evaluate] INFO | Metrics already computed: {out_file}")
                continue
            print(f"[evaluate] INFO | Running evaluation for {domain} and variable {var}")
            cmd = [
                "ml-eval", "benchmark",
                "--x0", str(x0),
                "--x1", str(x1),
                "--var", var,
                "--output", str(out_file),
                "--max-members", str(config.max_members),
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[evaluate] WARNING | Failed ({result.returncode}): {' '.join(cmd)}")
                n_failed += 1

    return n_failed


def aggregate_experiment(
    config: Config,
    train_exp: str,
    test_exp: str,
    keep_domain_files: bool = False,
) -> None:
    """
    Aggregate the per-domain metrics of one experiment pair into a summary.

    Reads `metrics_{var}_{domain}.json` and writes `metrics_summary.json` in the
    same directory, holding for each variable and metric the aggregation rule,
    the aggregate and the per-domain values.

    Missing data are not dropped: a NaN domain value makes the aggregate NaN
    (null in the JSON). Domains listed in the configuration without a metric file
    are reported, and the aggregate covers only the domains found. If no metric
    file is found, nothing is written and an existing summary is left untouched.
    `n_members` (members in the predictions) and `members` (positions of the
    members used; null for deterministic models) are null if the domain files
    disagree or lack them (files written by older versions of `ml-eval`).

    Parameters
    ----------
    config : Config
        Model, domains and variables to aggregate.
    train_exp : str
        Training experiment, e.g. `ESD_pseudo_reality`.
    test_exp : str
        Test experiment, e.g. `Perfect_cross_validation`.
    keep_domain_files : bool, default False
        Keep the per-domain metric files after the summary is written. If False,
        they are deleted, and a later `--aggregate-only` run has nothing to read
        for this experiment.
    """
    out_dir = experiment_dir(config, train_exp, test_exp)

    summary = {
        "model": config.model,
        "train_experiment": train_exp,
        "test_experiment": test_exp,
        "ensemble": {
            "n_members": None,
            "members": None,
        },
        "variables": {},
    }
    src_files = []
    n_members, members = [], []
    for var in config.variables:
        # read domain-specific metrics
        results = {}
        for domain in config.domains:
            file = metrics_file(out_dir, var, domain)
            if not file.is_file():
                print(f"[aggregate] WARNING | Missing file: {file}")
                continue
            with open(file) as f:
                results[domain] = json.load(f)
            # extract ensemble information
            attrs = results[domain].pop("attrs", {})
            n_members.append(attrs.get("n_members"))
            members.append(attrs.get("members"))
            src_files.append(file)
        if not results:
            continue
        domains = list(results)

        # aggregate metrics across domains
        metric_names = dict.fromkeys(m for values in results.values() for m in values)
        metrics = {}
        for metric in metric_names:
            func = AGGREGATION.get(metric)
            if func is None:
                print(
                    f"[aggregate] WARNING | No aggregation rule for {var} metric {metric!r}"
                )
                continue
            values = np.array(
                [results[d].get(metric, np.nan) for d in domains], dtype=float
            )
            metrics[metric] = {
                "aggregation": func.__name__,
                "aggregate": _to_json_float(func(values)),
                "domains": {d: _to_json_float(v) for d, v in zip(domains, values)},
            }
        summary["variables"][var] = metrics

    if not summary["variables"]:
        print(f"[aggregate] WARNING | No metric files in {out_dir}")
        return
    summary["ensemble"]["n_members"] = _check_attributes(n_members, "n_members", out_dir)
    summary["ensemble"]["members"] = _check_attributes(members, "members", out_dir)

    # export
    summary_file = out_dir / SUMMARY_FILE
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] INFO | Summary saved: {summary_file}")

    # only after the summary is safely written
    if not keep_domain_files:
        for file in src_files:
            file.unlink()


def main() -> None:
    """Read the configuration file, then evaluate and aggregate each experiment."""
    # keep prints in order with the output of `ml-eval` when logging to a file
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    config = Config.from_yaml(args.config_file)
    dataset_mapping = load_yaml(BENCHMARK_DIR / "datasets.yaml")
    experiment_mapping = load_yaml(BENCHMARK_DIR / "experiments.yaml")

    n_failed = 0
    for train_exp, test_exps in config.experiments.items():
        for test_exp in test_exps:
            print(f"[main] INFO | Train/test experiments: {train_exp} / {test_exp}")
            summary_file = experiment_dir(config, train_exp, test_exp) / SUMMARY_FILE
            attrs = experiment_mapping[train_exp]['tests'][test_exp]
            if not args.aggregate_only:
                if summary_file.is_file() and not args.overwrite:
                    print(f"[main] INFO | Summary already computed: {summary_file}")
                    continue
                n_failed += evaluate_experiment(
                    config,
                    train_exp,
                    test_exp,
                    attrs,
                    dataset_mapping,
                    overwrite=args.overwrite,
                )

            aggregate_experiment(
                config, train_exp, test_exp, keep_domain_files=args.keep_domain_files
            )

    if n_failed > 0:
        sys.exit(f"{n_failed} evaluation(s) failed")


if __name__ == "__main__":
    main()
