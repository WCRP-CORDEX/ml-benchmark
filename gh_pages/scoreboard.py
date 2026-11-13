#!/usr/bin/env python
"""
Build the scoreboard tables of the docs site from the models' metric summaries.

Reads `data/<MODEL>/evaluation/<TRAIN_EXP>/<TEST_EXP>/metrics_summary.json` (written
by `evaluate.py`) for every model and writes one page per training experiment,
`docs/results/scoreboard/<train_exp>.md` (lower case), with one HTML table per
variable (rows = models, columns = model type and metrics; each cell shows the
values of the experiment's tests, separated by a slash). The tables go between the
markers `<!-- SCOREBOARD:START -->` and `<!-- SCOREBOARD:END -->`, which each page
must already contain. On the docs site, the tables are sortable by clicking a
header (docs/javascripts/scoreboard.js).

Usage
-----
    ./scoreboard.py            # update the pages in docs/results/scoreboard/
    ./scoreboard.py --print    # print the tables to stdout instead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
BENCHMARK_DIR = ROOT_DIR / "benchmark"
OUTPUT_DIR = ROOT_DIR / "docs" / "results" / "scoreboard"  # one page per training exp.
SUMMARY_FILE = "metrics_summary.json"
METADATA_FILE = "metadata.yaml"
OROG_SUFFIX = "-orog"
START_MARKER = "<!-- SCOREBOARD:START -->"
END_MARKER = "<!-- SCOREBOARD:END -->"

# training experiment -> (label, {test experiment: label}), in display order. Each
# training experiment has its own page; each cell shows the values of its tests,
# in this order (left to right)
TRAINING_GROUPS = {
    "ESD_pseudo_reality": (
        "ESD",
        {
            "Perfect_cross_validation": "Perfect cross-validation",
            "Perfect_extrapolation": "Perfect extrapolation",
        },
    ),
    "Emulator_hist_future": (
        "Emulator",
        {
            "Perfect_cross_validation": "Perfect cross-validation",
            "Perfect_interpolation": "Perfect interpolation",
        },
    ),
}

# variable -> {metric key in metrics_summary.json: (label, higher_is_better)}
METRICS = {
    "tasmax": {
        "rmse": ("RMSE", False),
        "bias_mean": ("Clim. Mean", False),
        "bias_max1day": ("TXx", False),
        "ralsd": ("RALSD", False),
        "pss": ("PSS", True),
        "bias_annstd": ("IAV", False),
    },
    "pr": {
        "rmse": ("RMSE", False),
        "bias_mean": ("Clim. Mean", False),
        "bias_sdii": ("SDII", False),
        "bias_max1day": ("Rx1day", False),
        "ralsd": ("RALSD", False),
        "lhd": ("LHD", False),
    },
}

# `stochastic_output` in data/<MODEL>/metadata.yaml -> model type; the scoreboard
# shows one group of rows per type, in this order (models without metadata last)
MODEL_TYPES = {
    False: "Deterministic",
    True: "Generative",
}
UNKNOWN_TYPE = "Type not registered"

# model type -> (short label in the Type column, CSS modifier of its badge); the
# colours are in docs/stylesheets/extra.css
TYPE_BADGES = {
    "Deterministic": ("Det.", "deterministic"),
    "Generative": ("Gen.", "generative"),
    UNKNOWN_TYPE: ("?", "unknown"),
}

VARIABLE_LABELS = {
    "tasmax": "Maximum temperature",
    "pr": "Total precipitation",
}

LEGEND = """\
Metrics are computed per domain by `ml-eval benchmark` and aggregated across
domains. Biases are spatial means of the absolute bias, in the units of the variable. 
Lower is better for all metrics except PSS. Each cell shows  the two tests of the 
training experiment, separated by a slash in the order given in  the column header; 
the best value of each test is in bold, and – marks results not  available. 
Models using orography as a predictor are in italics. Type: 
Det. = deterministic, Gen. = generative.

Click a metric to sort the models, best first: the first click sorts by the first test,
the next one by the second test; missing values always go last. Click "Type" to reset
the original sorting.
"""


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Arguments, with the attribute `print`.
    """
    parser = argparse.ArgumentParser(
        description="Build the scoreboard pages of docs/results/scoreboard/."
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="print the tables to stdout instead of updating the pages",
    )
    return parser.parse_args()


def collect(data_dir: Path = DATA_DIR) -> dict[str, dict[tuple[str, str], dict]]:
    """
    Read the metric summaries of all models.

    Parameters
    ----------
    data_dir : Path
        Directory holding one sub-directory per model.

    Returns
    -------
    dict[str, dict[tuple[str, str], dict]]
        {model: {(training experiment, test experiment): summary}}, models sorted
        alphabetically by name (case-insensitive).
    """
    results: dict[str, dict[tuple[str, str], dict]] = {}
    for file in data_dir.glob(f"*/evaluation/*/*/{SUMMARY_FILE}"):
        test_exp = file.parent.name
        train_exp = file.parent.parent.name
        model = file.parents[3].name
        with open(file) as f:
            results.setdefault(model, {})[(train_exp, test_exp)] = json.load(f)
    return dict(sorted(results.items(), key=lambda item: item[0].lower()))


def model_type(model: str, data_dir: Path = DATA_DIR) -> str:
    """
    Type of a model (a value of `MODEL_TYPES`), from its metadata.

    Parameters
    ----------
    model : str
        Model directory name under `data_dir`.
    data_dir : Path
        Directory holding one sub-directory per model.

    Returns
    -------
    str
        Value of `MODEL_TYPES` for its `stochastic_output`, or `UNKNOWN_TYPE` if the
        model has no metadata file or the field is missing/null.
    """
    file = data_dir / model / METADATA_FILE
    if not file.exists():
        return UNKNOWN_TYPE
    metadata = yaml.safe_load(file.read_text()) or {}
    return MODEL_TYPES.get(metadata.get("stochastic_output"), UNKNOWN_TYPE)


def _format(value: float | None) -> str:
    """Format a metric value with 2 decimals; None -> en dash."""
    if value is None:
        return "–"
    return f"{value:.2f}"


def _domains(summary: dict, var: str) -> set[str]:
    """Domains covered by the aggregates of one variable in a summary."""
    metrics = summary.get("variables", {}).get(var, {})
    return {domain for metric in metrics.values() for domain in metric["domains"]}


def build_table(
    results: dict[str, dict[tuple[str, str], dict]],
    var: str,
    all_domains: list[str],
    types: dict[str, str],
    train_exp: str,
) -> str:
    """
    Build the HTML table of one variable and one training experiment.

    Rows are grouped by model type (deterministic, generative, then models without
    metadata), shown by a badge in a Type column, and sorted alphabetically within
    a type; the best values are marked over all models. Models using orography
    (`-orog` suffix) are in italics. Missing experiments, metrics or null
    aggregates are shown as an en dash; they are never filled. Models whose
    aggregates do not cover all the benchmark domains in some experiment are
    marked with a dagger and listed, with the experiment and the domains used, in
    a footnote.

    Parameters
    ----------
    results : dict
        Output of `collect`.
    var : str
        Variable, a key of `METRICS`.
    all_domains : list[str]
        All benchmark domains, from `benchmark/datasets.yaml`.
    types : dict[str, str]
        Type of each model in `results` (see `model_type`).
    train_exp : str
        Training experiment, a key of `TRAINING_GROUPS`. Only its tests are shown,
        and only models with results in at least one of them get a row.

    Returns
    -------
    str
        HTML table followed by its footnotes (if any).
    """
    metrics = METRICS[var]
    _, tests = TRAINING_GROUPS[train_exp]
    experiments = [(train_exp, test_exp) for test_exp in tests]

    # values[model][(experiment, metric)] = aggregate or None
    values: dict[str, dict[tuple, float | None]] = {}
    partial: dict[str, list[str]] = {}  # model -> footnote entries
    for model, summaries in results.items():
        row = {}
        for exp in experiments:
            summary = summaries.get(exp)
            var_metrics = (summary or {}).get("variables", {}).get(var, {})
            for metric in metrics:
                row[(exp, metric)] = var_metrics.get(metric, {}).get("aggregate")
            if summary is not None:
                domains = _domains(summary, var)
                if domains and set(all_domains) - domains:
                    exp_label = tests[exp[1]]
                    used = ", ".join(d for d in all_domains if d in domains)
                    partial.setdefault(model, []).append(f"{exp_label}: {used}")
        if any(v is not None for v in row.values()):
            values[model] = row

    # models per type, in display order (alphabetical within a type)
    groups: dict[str, list[str]] = {}
    for group in [*MODEL_TYPES.values(), UNKNOWN_TYPE]:
        members = [model for model in values if types[model] == group]
        if members:
            groups[group] = members

    # best value per test and metric, over all models
    best = {}
    for exp in experiments:
        for metric, (_, higher_is_better) in metrics.items():
            column = [row[(exp, metric)] for row in values.values()]
            column = [v for v in column if v is not None]
            if column:
                best[(exp, metric)] = max(column) if higher_is_better else min(column)

    # The wrapper's class lets docs/stylesheets/extra.css style these tables only,
    # and docs/javascripts/scoreboard.js make them sortable. Sortable headers carry
    # `data-col` (column index in the body) and, for metrics, `data-best` (sort
    # direction that puts the best values first); body cells carry the exact
    # values as `data-sort` (type) or `data-values` (one per test, ";"-separated,
    # empty if missing). GitHub ignores the classes and data attributes.
    lines = [
        '<div class="scoreboard">',
        f'<table data-tests="{";".join(tests.values())}">',
        "  <thead>",
        "    <tr>",
        # empty corner cell over the model names
        '      <th rowspan="2"></th>',
        '      <th rowspan="2" class="sortable" data-col="1">Type</th>',
    ]
    # the order in which each cell shows the tests; the training experiment is
    # the page itself
    order = " / ".join(tests.values())
    lines.append(f'      <th colspan="{len(metrics)}">{order}</th>')
    lines += ["    </tr>", "    <tr>"]
    for col, (label, higher_is_better) in enumerate(metrics.values(), start=2):
        direction = "desc" if higher_is_better else "asc"
        lines.append(
            f'      <th class="sortable" data-col="{col}" data-best="{direction}">'
            f"{label}</th>"
        )
    lines += ["    </tr>", "  </thead>", "  <tbody>"]
    # rows grouped by type (in `groups` order), alphabetical within a type; the
    # type is shown by a coloured badge in its own column
    for rank, (group, members) in enumerate(groups.items()):
        badge, modifier = TYPE_BADGES[group]
        for model in members:
            row = values[model]
            name = f"<i>{model}</i>" if model.endswith(OROG_SUFFIX) else model
            if model in partial:
                name += "<sup>†</sup>"
            lines += [
                f'    <tr class="model-type--{modifier}">',
                f"      <td>{name}</td>",
                f'      <td data-sort="{rank}" title="{group}">'
                f'<span class="model-type">{badge}</span></td>',
            ]
            for metric in metrics:
                cell, exact = [], []
                for exp in experiments:
                    key = (exp, metric)
                    text = _format(row[key])
                    if row[key] is not None and row[key] == best.get(key):
                        text = f"<b>{text}</b>"
                    cell.append(text)
                    exact.append("" if row[key] is None else repr(row[key]))
                lines.append(
                    f'      <td align="right" data-values="{";".join(exact)}">'
                    f'{" / ".join(cell)}</td>'
                )
            lines.append("    </tr>")
    lines += ["  </tbody>", "</table>", "</div>"]

    if partial:
        lines.append("")
        notes = "; ".join(
            f"{model}, {entry}"
            for model, entries in partial.items()
            for entry in entries
        )
        lines.append(f"† Aggregated over a subset of domains only ({notes}).")
    return "\n".join(lines)


def render(results: dict[str, dict[tuple[str, str], dict]], train_exp: str) -> str:
    """
    Render the scoreboard block of one training experiment.

    Parameters
    ----------
    results : dict
        Output of `collect`.
    train_exp : str
        Training experiment, a key of `TRAINING_GROUPS`.

    Returns
    -------
    str
        Markdown/HTML text to place between the scoreboard markers of the page:
        the legend and one table per variable, each under a `###` heading.
    """
    with open(BENCHMARK_DIR / "datasets.yaml") as f:
        all_domains = list(yaml.safe_load(f))

    types = {model: model_type(model) for model in results}
    parts = [LEGEND]
    for var in METRICS:
        parts.append(f"### {VARIABLE_LABELS[var]}\n")
        # blank lines around the HTML so GitHub renders the Markdown that follows
        parts.append(build_table(results, var, all_domains, types, train_exp) + "\n")
    return "\n".join(parts)


def update_block(
    path: Path,
    block: str,
    start_marker: str = START_MARKER,
    end_marker: str = END_MARKER,
) -> bool:
    """
    Replace the text between two markers of a markdown file.

    Parameters
    ----------
    path : Path
        Markdown file containing `start_marker` and `end_marker`.
    block : str
        New content between the markers.
    start_marker, end_marker : str
        Markers delimiting the block; by default, those of the scoreboard.

    Returns
    -------
    bool
        True if the file changed.

    Raises
    ------
    ValueError
        If the markers are missing or in the wrong order.
    """
    text = path.read_text()
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < start:
        raise ValueError(f"Markers {start_marker} / {end_marker} not found in {path}")
    new_text = (
        text[: start + len(start_marker)] + "\n\n" + block.strip() + "\n\n" + text[end:]
    )
    if new_text == text:
        return False
    path.write_text(new_text)
    return True


def output_file(train_exp: str) -> Path:
    """Scoreboard page of a training experiment."""
    return OUTPUT_DIR / f"{train_exp.lower()}.md"


def main() -> None:
    """Collect the metric summaries and print or write the scoreboard pages."""
    args = parse_args()
    results = collect()
    for train_exp, (train_label, _) in TRAINING_GROUPS.items():
        block = render(results, train_exp)
        page = output_file(train_exp)
        if args.print:
            # one top-level section per training experiment (e.g. job summary)
            print(f"# {train_label}\n\n{block}\n")
        elif update_block(page, block):
            print(f"[scoreboard] INFO | Scoreboard updated: {page}")
        else:
            print(f"[scoreboard] INFO | Scoreboard already up to date: {page}")


if __name__ == "__main__":
    main()
