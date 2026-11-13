#!/usr/bin/env python
"""
Build the tables of contributing models of the docs site from the models'
metadata.

Reads `data/<MODEL>/metadata.yaml` (see `submission/metadata_template.yaml`) for
every model and writes one HTML table per model type (Model | Description |
Submission | Reference | Implementation) between the markers `<!-- MODELS:START -->` and
`<!-- MODELS:END -->` in `docs/results/models.md`.

A model and its orography variant (`<MODEL>-orog`) share one row, named after the
model without the suffix and filled from the metadata of the version without
orography when both exist. Models are grouped by type, from `stochastic_output`
(deterministic, generative, then models whose type is not registered): each group
is a section with its own heading and table, sorted alphabetically
(case-insensitive). Groups without models are omitted.

Usage
-----
    python submission/models_table.py            # update docs/results/models.md
    python submission/models_table.py --print    # print the table to stdout instead
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_FILE = ROOT_DIR / "docs" / "results" / "models.md"
METADATA_FILE = "metadata.yaml"
START_MARKER = "<!-- MODELS:START -->"
END_MARKER = "<!-- MODELS:END -->"
OROG_SUFFIX = "-orog"

# table columns, in display order
COLUMNS = ("Model", "Description", "Submission", "Reference", "Implementation")

# text shown for a missing reference or implementation
MISSING = "In preparation"

# value of `blind` -> text of the Submission column; null shows UNKNOWN_RELEASE
RELEASE_LABELS = {True: "Pre-release", False: "Post-release"}
UNKNOWN_RELEASE = "Unknown"

# host of the implementation URL -> link text; any other host shows "Link"
HOST_LABELS = {
    "github.com": "GitHub",
    "gitlab.com": "GitLab",
    "zenodo.org": "Zenodo",
}

# metadata fields that must be present (they may be null)
REQUIRED_FIELDS = (
    "emulator_id",
    "short_description",
    "reference_label",
    "reference_url",
    "reproducibility_url",
    "stochastic_output",
    "blind",
)

# sys.path is extended so the helper shared with the scoreboard can be reused
sys.path.insert(0, str(ROOT_DIR / "evaluation"))
from scoreboard import MODEL_TYPES, UNKNOWN_TYPE, update_block  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Arguments, with the attribute `print`.
    """
    parser = argparse.ArgumentParser(
        description="Build the table of contributing models of docs/results/models.md."
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="print the table to stdout instead of updating the file",
    )
    return parser.parse_args()


def collect(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Read the metadata of every model.

    Parameters
    ----------
    data_dir : Path
        Directory with one subdirectory per model.

    Returns
    -------
    list of dict
        Metadata of each model, with the key `model` set to its directory name.

    Raises
    ------
    ValueError
        If a metadata file lacks a required field.
    """
    models = []
    for path in sorted(data_dir.glob(f"*/{METADATA_FILE}")):
        metadata = yaml.safe_load(path.read_text()) or {}
        missing = [field for field in REQUIRED_FIELDS if field not in metadata]
        if missing:
            raise ValueError(f"{path} lacks the field(s): {', '.join(missing)}")
        metadata["model"] = path.parent.name
        models.append(metadata)
    return models


def merge_orography(models: list[dict]) -> list[dict]:
    """
    Keep one entry per model, merging each `<MODEL>-orog` with `<MODEL>`.

    The entry of the version without orography is kept when both exist.

    Parameters
    ----------
    models : list of dict
        Metadata as returned by `collect`.

    Returns
    -------
    list of dict
        Metadata with the key `name` (model name without the orography suffix).
    """
    merged: dict[str, dict] = {}
    for metadata in models:
        name = metadata["model"].removesuffix(OROG_SUFFIX)
        is_orog = metadata["model"] != name
        # the version without orography fills the row when both exist
        if name not in merged or not is_orog:
            merged[name] = {**metadata, "name": name}
    return list(merged.values())


def _text(value: str) -> str:
    """Make a text safe for an HTML table cell (whitespace collapsed)."""
    return html.escape(" ".join(str(value).split()), quote=False)


def _link(label: str, url: str) -> str:
    """HTML link."""
    return f'<a href="{html.escape(url, quote=True)}">{_text(label)}</a>'


def _reference(
    label: str | list[str] | None, url: str | list[str] | None
) -> str:
    """
    Render the reference column.

    `label` and `url` may be lists (several references), paired by position and
    separated by "; " in the table.
    """
    if isinstance(label, list) or isinstance(url, list):
        labels = label if isinstance(label, list) else [label]
        urls = url if isinstance(url, list) else [url]
        if len(labels) != len(urls):
            raise ValueError(
                f"reference_label {labels} and reference_url {urls} differ in length"
            )
        return "; ".join(_reference(lab, u) for lab, u in zip(labels, urls))
    if label is None:
        return _link("Link", url) if url else MISSING
    return _link(label, url) if url else _text(label)


def _implementation(url: str | None) -> str:
    """Render the implementation column, labelling the link by its host."""
    if not url:
        return MISSING
    host = urlparse(url).netloc.lower().removeprefix("www.")
    label = HOST_LABELS.get(host, "GitLab" if host.startswith("gitlab.") else "Link")
    return _link(label, url)


def _release(blind: bool | None) -> str:
    """
    Render the Submission column from the `blind` field.

    Parameters
    ----------
    blind : bool or None
        True if the predictions were submitted before the release of the test
        targets, False if after, None if unknown.

    Returns
    -------
    str
        "Pre-release", "Post-release" or "Unknown".

    Raises
    ------
    ValueError
        If `blind` is not a boolean or null (e.g. the string "false").
    """
    if blind is None:
        return UNKNOWN_RELEASE
    if not isinstance(blind, bool):
        raise ValueError(f"blind must be true, false or null, not {blind!r}")
    return RELEASE_LABELS[blind]


def _table(models: list[dict]) -> str:
    """
    Render one HTML table of models.

    Parameters
    ----------
    models : list of dict
        Metadata of the models in the table (orography variants merged), in
        display order.

    Returns
    -------
    str
        HTML table with one row per model.
    """
    # the wrapper's class lets docs/stylesheets/extra.css style these tables only
    lines = ['<div class="models-table">', "<table>", "  <thead>", "    <tr>"]
    lines += [f"      <th>{column}</th>" for column in COLUMNS]
    lines += ["    </tr>", "  </thead>", "  <tbody>"]
    for m in models:
        cells = [
            _text(m["name"]),
            _text(m["short_description"] or ""),
            _release(m["blind"]),
            _reference(m["reference_label"], m["reference_url"]),
            _implementation(m["reproducibility_url"]),
        ]
        lines.append("    <tr>")
        lines += [f"      <td>{cell}</td>" for cell in cells]
        lines.append("    </tr>")
    lines += ["  </tbody>", "</table>", "</div>"]
    return "\n".join(lines)


def render(models: list[dict]) -> str:
    """
    Render the tables of contributing models, one section per model type.

    Parameters
    ----------
    models : list of dict
        Metadata as returned by `collect`.

    Returns
    -------
    str
        Markdown with, for each model type (`MODEL_TYPES`, then `UNKNOWN_TYPE`), a
        `## <type>` heading followed by an HTML table with one row per model
        (orography variants merged). Types without models are omitted.
    """
    rows = sorted(merge_orography(models), key=lambda m: m["name"].lower())
    sections = []
    for group in [*MODEL_TYPES.values(), UNKNOWN_TYPE]:
        members = [
            m for m in rows
            if MODEL_TYPES.get(m["stochastic_output"], UNKNOWN_TYPE) == group
        ]
        if members:
            sections.append(f"## {group}\n\n{_table(members)}")
    return "\n\n".join(sections)


def main() -> None:
    """Collect the models' metadata and print or write the table."""
    args = parse_args()
    block = render(collect())
    if args.print:
        print(block)
    elif update_block(OUTPUT_FILE, block, START_MARKER, END_MARKER):
        print(f"[models_table] INFO | Models table updated: {OUTPUT_FILE}")
    else:
        print(f"[models_table] INFO | Models table already up to date: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
