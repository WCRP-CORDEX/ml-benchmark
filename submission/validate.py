#!/usr/bin/env python
"""
Check that a submission complies with the CORDEX ML-Bench file structure.

The list of required prediction files is built from the benchmark definitions in
`benchmark/` (`experiments.yaml`, `datasets.yaml`, `horizons.yaml`), so it stays
in sync with `evaluate.py`. A submission must predict every test predictor, i.e.
every combination of domain, training experiment, horizon, case (perfect /
imperfect) and driving GCM (training / transfer GCM): 72 files for the full
benchmark. Every file must exist at

    <ROOT>/<DOMAIN>_Domain/<TRAIN_EXP>/<horizon>/<case>/Predictions_pr_tasmax_<GCM>_<period>.nc

and is checked against the templates in `format_predictions/templates/`:

- the file opens and its data can be read (first value of each variable);
- it contains both `pr` and `tasmax`;
- their dimensions are `time` plus the template's spatial dimensions, and
  optionally `member` (ensemble of generative models, at least `--min-members`);
- `time` and the spatial dimensions are in the template's order (e.g.
  `(time, y, x)`, not `(time, x, y)`): a different order makes the evaluation
  transpose the data, which can read whole files into memory;
- the spatial grid (size and dimension coordinates) equals the template's, and
  each spatial dimension has a 1-D (index) coordinate, not e.g. a 2-D `y(y, x)`;
- the time axis contains every year of `<period>` (missing years are errors,
  years outside the period warnings).

Extra `.nc` files that match no test predictor are reported as warnings,
since they usually point to a misnamed or misplaced file. The data values
themselves (e.g. NaN, units, plausible ranges) are not checked.

The submission can be a directory or a `.zip` file.

Usage
-----
From the root of the repository (`pixi run validate ...` is equivalent):

    python submission/validate.py models/<MODEL>/predictions
    python submission/validate.py <MODEL>.zip
    python submission/validate.py models/<MODEL>/predictions --domains ALPS NZ
    python submission/validate.py models/<MODEL>/predictions --experiments ESD_pseudo_reality
    python submission/validate.py models/<MODEL>/predictions --no-content  # structure only

The exit code is 1 if any error is found, 0 otherwise.
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT_DIR / "benchmark"
TEMPLATES_DIR = ROOT_DIR / "format_predictions" / "templates"

VARIABLES = ("pr", "tasmax")
TIME_DIM = "time"
MEMBER_DIM = "member"
MIN_MEMBERS = 5  # minimum ensemble size required for generative models


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    """One validation finding."""

    severity: str  # "ERROR" | "WARNING"
    code: str  # stable identifier, e.g. MISSING_FILE
    message: str
    file: str = ""


@dataclass
class ExpectedFile:
    """One prediction file required by the benchmark."""

    domain: str
    train_exp: str
    horizon: str
    case: str
    gcm: str
    period: str
    tests: list[str] = field(default_factory=list)

    @property
    def relative_path(self) -> Path:
        """Path of the file relative to the submission root."""
        filename = f"Predictions_pr_tasmax_{self.gcm}_{self.period}.nc"
        return (
            Path(f"{self.domain}_Domain") / self.train_exp / self.horizon
            / self.case / filename
        )


@dataclass
class Report:
    """Validation outcome for one submission."""

    root: Path
    issues: list[Issue] = field(default_factory=list)
    n_expected: int = 0
    n_checked: int = 0

    def add(self, severity: str, code: str, message: str, file: Path | str = "") -> None:
        """Append an issue to the report."""
        self.issues.append(Issue(severity, code, message, str(file)))

    @property
    def n_errors(self) -> int:
        """Number of ERROR issues."""
        return sum(issue.severity == "ERROR" for issue in self.issues)

    @property
    def n_warnings(self) -> int:
        """Number of WARNING issues."""
        return sum(issue.severity == "WARNING" for issue in self.issues)


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)


def expand_horizon(horizon: str, horizons: dict[str, str]) -> list[str]:
    """
    Expand a test horizon into the concrete horizons with prediction files.

    A horizon whose period in `horizons.yaml` contains a wildcard (e.g.
    `mid_end_century: '*-*'`) covers every concrete horizon ending with the same
    suffix (`*_century` -> `mid_century`, `end_century`), the same rule that
    `evaluate.py` uses to build its paths.

    Parameters
    ----------
    horizon : str
        Horizon of a test in `experiments.yaml`.
    horizons : dict[str, str]
        Mapping horizon -> period from `horizons.yaml`.

    Returns
    -------
    list[str]
        Concrete horizons (with a `START-END` period).
    """
    if "*" not in str(horizons[horizon]):
        return [horizon]
    pattern = "*_" + horizon.rsplit("_", 1)[-1]
    expanded = [
        name for name, period in horizons.items()
        if "*" not in str(period) and fnmatch.fnmatch(name, pattern)
    ]
    if not expanded:
        raise ValueError(f"Wildcard horizon '{horizon}' matches no concrete horizon.")
    return expanded


def expected_files(
    experiments: dict,
    datasets: dict,
    horizons: dict[str, str],
    domains: list[str],
    train_exps: list[str],
) -> list[ExpectedFile]:
    """
    List the prediction files required for the given domains and experiments.

    A submission must predict every test predictor, so a file is required for
    each combination of domain, training experiment, horizon , case and driving
    GCM (72 files) for the full benchmark. Each file records the test 
    experiments that score it.

    Parameters
    ----------
    experiments, datasets, horizons : dict
        Contents of `experiments.yaml`, `datasets.yaml` and `horizons.yaml`.
    domains : list[str]
        Domains to include.
    train_exps : list[str]
        Training experiments to include.

    Returns
    -------
    list[ExpectedFile]
        Required files, in a stable order.
    """
    concrete_horizons = [
        name for name, period in horizons.items() if "*" not in str(period)
    ]
    tests = [attrs for exp in experiments.values() for attrs in exp["tests"].values()]
    cases = sorted({attrs["case"] for attrs in tests}, reverse=True)

    # every combination of domain, training experiment, horizon, case and GCM
    files = {}
    for domain in domains:
        gcms = (datasets[domain]["train_gcm"], datasets[domain]["test_gcm"])
        for train_exp in train_exps:
            for horizon in concrete_horizons:
                for case in cases:
                    for gcm in gcms:
                        item = ExpectedFile(
                            domain=domain,
                            train_exp=train_exp,
                            horizon=horizon,
                            case=case,
                            gcm=gcm,
                            period=str(horizons[horizon]),
                        )
                        files[item.relative_path] = item

    # tests that score each file
    for domain in domains:
        for train_exp in train_exps:
            for test_exp, attrs in experiments[train_exp]["tests"].items():
                gcm_key = "test_gcm" if attrs["transfer"] else "train_gcm"
                for horizon in expand_horizon(attrs["horizon"], horizons):
                    item = ExpectedFile(
                        domain=domain,
                        train_exp=train_exp,
                        horizon=horizon,
                        case=attrs["case"],
                        gcm=datasets[domain][gcm_key],
                        period=str(horizons[horizon]),
                    )
                    files[item.relative_path].tests.append(test_exp)
    return list(files.values())


# ---------------------------------------------------------------------------
# Submission root
# ---------------------------------------------------------------------------


def find_root(path: Path) -> Path:
    """
    Return the folder that contains the `<DOMAIN>_Domain` directories.

    If `path` does not contain them but holds a single sub-directory (e.g.
    `<emulator_id>/`), that sub-directory is used instead.
    """
    if any(path.glob("*_Domain")):
        return path
    subdirs = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(subdirs) == 1 and any(subdirs[0].glob("*_Domain")):
        return subdirs[0]
    return path


# ---------------------------------------------------------------------------
# File checks
# ---------------------------------------------------------------------------

_templates: dict[tuple[str, str], xr.Dataset] = {}


def get_template(var: str, domain: str) -> xr.Dataset:
    """Load (once) the official template of a variable and domain."""
    key = (var, domain)
    if key not in _templates:
        _templates[key] = xr.load_dataset(TEMPLATES_DIR / f"{var}_{domain}.nc")
    return _templates[key]


def check_variable(
    ds: xr.Dataset,
    var: str,
    domain: str,
    min_members: int,
    path: Path,
    report: Report,
) -> None:
    """Check readability, dimensions and grid of one variable against its template."""
    da = ds[var]

    # readability: load a single value to force an actual read from disk
    try:
        da.isel({dim: 0 for dim in da.dims}).load()
    except Exception as exc:
        report.add("ERROR", "UNREADABLE_DATA", f"Cannot read '{var}': {exc}", path)
        return

    template = get_template(var, domain)
    spatial_dims = tuple(d for d in template[var].dims if d != TIME_DIM)

    # dimensions: time + template spatial dims (+ member)
    allowed = {TIME_DIM, MEMBER_DIM, *spatial_dims}
    missing = [d for d in (TIME_DIM, *spatial_dims) if d not in da.dims]
    unexpected = [d for d in da.dims if d not in allowed]
    if missing or unexpected:
        report.add(
            "ERROR", "WRONG_DIMENSIONS",
            f"'{var}' has dims {da.dims}; expected {(TIME_DIM, *spatial_dims)}"
            f" (+ optional '{MEMBER_DIM}').",
            path,
        )
    else:
        # order of time + spatial dims, as in the template (member may be anywhere)
        order = tuple(d for d in da.dims if d != MEMBER_DIM)
        expected_order = (TIME_DIM, *spatial_dims)
        if order != expected_order:
            report.add(
                "ERROR", "WRONG_DIMENSION_ORDER",
                f"'{var}' has dims in the order {order}; expected {expected_order}"
                f" as in the template (e.g. `ds.transpose(..., "
                f"{', '.join(repr(d) for d in expected_order)})`).",
                path,
            )
    if TIME_DIM in da.dims and da.sizes[TIME_DIM] == 0:
        report.add("ERROR", "EMPTY_TIME", f"'{var}' has an empty time axis.", path)
    if MEMBER_DIM in da.dims and da.sizes[MEMBER_DIM] < min_members:
        report.add(
            "ERROR", "TOO_FEW_MEMBERS",
            f"'{var}' has {da.sizes[MEMBER_DIM]} members; expected at least"
            f" {min_members}.",
            path,
        )

    # spatial grid: same size and dimension coordinates as the template
    for dim in spatial_dims:
        if dim not in da.dims:
            continue
        n_expected = template.sizes[dim]
        if da.sizes[dim] != n_expected:
            report.add(
                "ERROR", "WRONG_GRID_SIZE",
                f"'{var}': '{dim}' has size {da.sizes[dim]}; expected {n_expected}.",
                path,
            )
        elif dim not in ds.coords:
            report.add(
                "ERROR", "MISSING_COORDINATE",
                f"'{var}': no coordinate values for dimension '{dim}'.",
                path,
            )
        elif dim not in ds.indexes:
            # e.g. `y`/`x` stored as 2-D copies of lat/lon: xarray cannot
            # combine such files along time with `open_mfdataset`
            report.add(
                "ERROR", "NON_INDEX_COORDINATE",
                f"'{var}': coordinate '{dim}' has dims {ds[dim].dims}; expected a"
                f" 1-D coordinate ('{dim}',) as in the template.",
                path,
            )
        elif not np.allclose(ds[dim].values, template[dim].values):
            report.add(
                "ERROR", "WRONG_GRID_COORDINATES",
                f"'{var}': '{dim}' coordinates differ from the template.",
                path,
            )

    # auxiliary 2-D lat/lon (rotated/projected grids, e.g. ALPS): useful for maps
    for coord in ("lat", "lon"):
        if coord in template.coords and coord not in ds.variables:
            report.add(
                "WARNING", "MISSING_LAT_LON",
                f"'{coord}' coordinate is missing (present in the template).",
                path,
            )


def check_time_coverage(
    ds: xr.Dataset, expected: ExpectedFile, path: Path, report: Report
) -> None:
    """Check that the time axis covers every year of the expected period."""
    if TIME_DIM not in ds.coords or ds.sizes.get(TIME_DIM, 0) == 0:
        return
    try:
        years = set(ds[TIME_DIM].dt.year.values.tolist())
    except (TypeError, AttributeError):
        report.add(
            "ERROR", "UNDECODABLE_TIME",
            "The time coordinate cannot be decoded as dates.",
            path,
        )
        return

    start, end = (int(year) for year in expected.period.split("-"))
    expected_years = set(range(start, end + 1))
    if expected_years - years:
        report.add(
            "ERROR", "INCOMPLETE_TIME",
            f"Missing years of {expected.period}: {sorted(expected_years - years)}.",
            path,
        )
    if years - expected_years:
        report.add(
            "WARNING", "EXTRA_YEARS",
            f"Years outside {expected.period}: {sorted(years - expected_years)}.",
            path,
        )


def check_file(
    path: Path, expected: ExpectedFile, min_members: int, report: Report
) -> None:
    """Run all content checks on one prediction file."""
    try:
        with xr.open_dataset(path) as ds:
            missing = [var for var in VARIABLES if var not in ds.data_vars]
            if missing:
                report.add(
                    "ERROR", "MISSING_VARIABLES",
                    f"Missing variable(s): {', '.join(missing)}.",
                    path,
                )
            for var in VARIABLES:
                if var in ds.data_vars:
                    check_variable(ds, var, expected.domain, min_members, path, report)
            check_time_coverage(ds, expected, path, report)
    except Exception as exc:
        report.add("ERROR", "UNREADABLE_FILE", f"Cannot open file: {exc}", path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    root: Path,
    files: list[ExpectedFile],
    check_content: bool = True,
    min_members: int = MIN_MEMBERS,
) -> Report:
    """
    Validate a submission directory against the required files.

    Parameters
    ----------
    root : Path
        Folder that contains the `<DOMAIN>_Domain` directories.
    files : list[ExpectedFile]
        Required files (see `expected_files`).
    check_content : bool, default True
        Open each file and check variables, dimensions, grid and time. If
        False, only the presence of the files is checked.
    min_members : int, default 5
        Minimum size of the `member` dimension, if present.

    Returns
    -------
    Report
        All issues found.
    """
    report = Report(root=root, n_expected=len(files))
    if not root.is_dir():
        report.add("ERROR", "MISSING_ROOT", "Submission directory not found.", root)
        return report

    for expected in files:
        path = root / expected.relative_path
        if not path.is_file():
            if expected.tests:
                scored = f"scored by {expected.train_exp}: {', '.join(expected.tests)}"
            else:
                scored = "not scored by any test"
            report.add(
                "ERROR", "MISSING_FILE",
                f"Test predictor without prediction ({scored}).",
                expected.relative_path,
            )
            continue
        report.n_checked += 1
        if check_content:
            check_file(path, expected, min_members, report)

    # files that match no test predictor, within the selected domains/experiments
    required = {expected.relative_path for expected in files}
    selected = {
        Path(f"{expected.domain}_Domain") / expected.train_exp for expected in files
    }
    for path in sorted(root.rglob("*.nc")):
        relative = path.relative_to(root)
        if relative in required:
            continue
        if any(relative.is_relative_to(prefix) for prefix in selected):
            report.add(
                "WARNING", "UNEXPECTED_FILE",
                "File matches no test predictor (misnamed or misplaced?).",
                relative,
            )
    return report


def print_report(report: Report, label: Path | str | None = None) -> None:
    """
    Print the issues grouped by severity, then a one-line summary.

    Parameters
    ----------
    report : Report
        Validation outcome.
    label : Path or str, optional
        Name of the submission in the summary (default: `report.root`).
    """
    for severity in ("ERROR", "WARNING"):
        for issue in report.issues:
            if issue.severity != severity:
                continue
            file = Path(issue.file)
            if file.is_relative_to(report.root):
                file = file.relative_to(report.root)
            location = f" | {file}" if issue.file else ""
            print(f"[validate] {severity} | {issue.code}{location}\n    {issue.message}")
    status = "VALID" if report.n_errors == 0 else "INVALID"
    print(
        f"[validate] {status} | {label or report.root}:"
        f" {report.n_checked}/{report.n_expected} required files found, {report.n_errors} error(s),"
        f" {report.n_warnings} warning(s)."
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args(experiments: dict, datasets: dict) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check that a submission complies with the ML-Bench file structure."
    )
    parser.add_argument(
        "submission", type=Path,
        help="Submission directory or .zip file (e.g. models/<MODEL>/predictions).",
    )
    parser.add_argument(
        "--domains", nargs="+", choices=list(datasets), default=list(datasets),
        help="Domains to validate (default: all).",
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=list(experiments), default=list(experiments),
        help="Training experiments to validate (default: all).",
    )
    parser.add_argument(
        "--min-members", type=int, default=MIN_MEMBERS,
        help=f"Minimum size of the '{MEMBER_DIM}' dimension (default: {MIN_MEMBERS}).",
    )
    parser.add_argument(
        "--no-content", action="store_true",
        help="Only check that the required files exist; do not open them.",
    )
    return parser.parse_args()


def main() -> None:
    experiments = load_yaml(BENCHMARK_DIR / "experiments.yaml")
    datasets = load_yaml(BENCHMARK_DIR / "datasets.yaml")
    horizons = load_yaml(BENCHMARK_DIR / "horizons.yaml")
    args = parse_args(experiments, datasets)

    files = expected_files(experiments, datasets, horizons, args.domains, args.experiments)

    with tempfile.TemporaryDirectory() as tmp_dir:
        if args.submission.suffix == ".zip" and args.submission.is_file():
            print(f"[validate] INFO | Extracting {args.submission}")
            with zipfile.ZipFile(args.submission) as zf:
                zf.extractall(tmp_dir)
            root = find_root(Path(tmp_dir))
        elif args.submission.is_dir():
            root = find_root(args.submission)
        else:
            sys.exit(f"[validate] ERROR | Not a directory or .zip file: {args.submission}")

        report = validate(root, files, not args.no_content, args.min_members)
        print_report(report, label=args.submission)

    sys.exit(1 if report.n_errors else 0)


if __name__ == "__main__":
    main()
