#!/usr/bin/env python
"""
Register an approved submission: write `data/<MODEL>/metadata.yaml` and
`data/<MODEL>/config.yaml` from the body of its submission issue.

Run by the workflow `.github/workflows/submission.yml` when a maintainer adds the
label `Approved submission` to a submission issue. The issue body is the markdown
that GitHub renders from the issue form (`.github/ISSUE_TEMPLATE/
ml_benchmark_submission.yaml`): one `### <field label>` heading per field followed
by its value, or `_No response_` for an empty optional field.

- Fields are matched to their `id` through the labels of the issue form, so the
  form and this script cannot drift apart. The keys and their order are those of
  `submission/metadata_template.yaml`.
- `stochastic_output` (`yes`/`no`) and `blind` (`Pre-release`/`Post-release`) are
  stored as true/false. Fields that the form does not ask for
  (`short_description`, `reference_label`) are written as null, to be filled by a
  maintainer in the pull request.
- `issue`, `submitted` (date the issue was opened) and `approved` are added;
  dates are in UTC.
- `config.yaml` is `evaluation/config_template.yaml` with `model: <MODEL>`.

Optionally, the script writes the comment that the workflow posts on the issue
(`--comment-file`) and the outputs `emulator_id` and `blind` for later workflow
steps (to `$GITHUB_OUTPUT`, when set).

Usage
-----
    ISSUE_BODY="$(cat body.md)" ISSUE_NUMBER=42 ISSUE_CREATED_AT=2026-11-13T10:00:00Z \\
        python submission/register.py --comment-file comment.md
    python submission/register.py --body-file body.md --issue 42 \\
        --created-at 2026-11-13 --data-dir /tmp/data      # local test

The exit code is 1 (with a message) if the issue is invalid: a required field is
empty, `emulator_id` is not a valid directory name, or `data/<MODEL>/` exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
FORM_FILE = ROOT_DIR / ".github" / "ISSUE_TEMPLATE" / "ml_benchmark_submission.yaml"
METADATA_TEMPLATE = ROOT_DIR / "submission" / "metadata_template.yaml"
CONFIG_TEMPLATE = ROOT_DIR / "evaluation" / "config_template.yaml"
DEFAULT_REPOSITORY = "WCRP-CORDEX/ml-benchmark"

NO_RESPONSE = "_No response_"
EMULATOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# form field id -> metadata key, when they differ
FIELD_ALIASES = {"emulator": "description"}

# metadata keys written by this script (the rest come from the form, or are null)
REGISTRATION_KEYS = ("issue", "submitted", "approved", "blind")

# answer of a dropdown -> boolean, by metadata key (matched on the first word)
BOOLEAN_ANSWERS = {
    "stochastic_output": {"yes": True, "no": False},
    "blind": {"pre-release": True, "post-release": False},
}


class RegistrationError(Exception):
    """The issue cannot be registered (message shown to the maintainers)."""


# ---------------------------------------------------------------------------
# Issue body
# ---------------------------------------------------------------------------


def load_form_fields(form_file: Path = FORM_FILE) -> dict[str, dict]:
    """
    Read the input fields of the issue form.

    Parameters
    ----------
    form_file : Path
        Issue form (GitHub issue-form YAML).

    Returns
    -------
    dict[str, dict]
        Field label -> {"id": field id, "required": bool}, for every field with an
        `id` (markdown blocks are skipped).
    """
    with open(form_file) as f:
        form = yaml.safe_load(f)
    fields = {}
    for item in form["body"]:
        if "id" not in item:
            continue
        label = item["attributes"]["label"]
        required = bool(item.get("validations", {}).get("required", False))
        fields[label] = {"id": item["id"], "required": required}
    return fields


def parse_issue_body(body: str, form_fields: dict[str, dict]) -> dict[str, str | None]:
    """
    Extract the answers of an issue form from the rendered issue body.

    Parameters
    ----------
    body : str
        Markdown body of the issue: `### <label>` headings followed by values.
    form_fields : dict[str, dict]
        Output of `load_form_fields`.

    Returns
    -------
    dict[str, str | None]
        Field id -> answer (stripped text; None if empty or `_No response_`).
        Headings that are not form labels are ignored.
    """
    answers: dict[str, str | None] = {}
    sections = re.split(r"^###[ \t]+", body.replace("\r\n", "\n"), flags=re.MULTILINE)
    for section in sections[1:]:  # sections[0] is the text before the first heading
        label, _, value = section.partition("\n")
        field = form_fields.get(label.strip())
        if field is None:
            continue
        value = value.strip()
        answers[field["id"]] = None if value in ("", NO_RESPONSE) else value
    return answers


def to_boolean(key: str, answer: str | None) -> bool | None:
    """Convert a dropdown answer (`yes`, `Pre-release (...)`, ...) to a boolean."""
    if answer is None:
        return None
    first_word = answer.split()[0].lower()
    try:
        return BOOLEAN_ANSWERS[key][first_word]
    except KeyError:
        raise RegistrationError(f"Unexpected answer for '{key}': {answer!r}") from None


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def build_metadata(
    answers: dict[str, str | None],
    form_fields: dict[str, dict],
    issue: int,
    submitted: dt.date,
    approved: dt.date,
) -> dict:
    """
    Build the metadata of a model from the answers of its submission issue.

    Parameters
    ----------
    answers : dict[str, str | None]
        Output of `parse_issue_body`.
    form_fields : dict[str, dict]
        Output of `load_form_fields` (to check the required fields).
    issue : int
        Number of the submission issue.
    submitted, approved : datetime.date
        Dates (UTC) the issue was opened and approved.

    Returns
    -------
    dict
        Metadata with the keys of `metadata_template.yaml`, in the same order.

    Raises
    ------
    RegistrationError
        If a required field is empty or `emulator_id` is not a valid name.
    """
    missing = [
        label for label, field in form_fields.items()
        if field["required"] and answers.get(field["id"]) is None
    ]
    if missing:
        raise RegistrationError(f"Required field(s) without answer: {', '.join(missing)}")

    emulator_id = answers["emulator_id"]
    if not EMULATOR_ID_PATTERN.match(emulator_id):
        raise RegistrationError(
            f"Invalid emulator_id {emulator_id!r}: use only letters, digits, '.', '_'"
            " and '-', starting with a letter or digit."
        )

    values = {FIELD_ALIASES.get(key, key): value for key, value in answers.items()}
    for key in BOOLEAN_ANSWERS:
        values[key] = to_boolean(key, values.get(key))
    values.update(issue=issue, submitted=submitted, approved=approved)

    with open(METADATA_TEMPLATE) as f:
        keys = list(yaml.safe_load(f))
    return {key: values.get(key) for key in keys}


def template_header(path: Path = METADATA_TEMPLATE) -> str:
    """
    Return the comment block at the top of the metadata template, without its
    last paragraph (which only describes the example values).
    """
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#") or "The values below are only an example" in line:
            break
        lines.append(line)
    while lines and lines[-1].strip() == "#":
        lines.pop()
    return "\n".join(lines) + "\n"


class _MetadataDumper(yaml.SafeDumper):
    """YAML dumper that writes multi-line strings as literal blocks (`|-`)."""


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_MetadataDumper.add_representer(str, _represent_str)


def _dump(values: dict) -> str:
    return yaml.dump(
        values, Dumper=_MetadataDumper, sort_keys=False, allow_unicode=True, width=88
    )


def metadata_text(metadata: dict) -> str:
    """Render the metadata file: template header, issue fields, registration fields."""
    issue_part = {k: v for k, v in metadata.items() if k not in REGISTRATION_KEYS}
    registration_part = {k: metadata[k] for k in REGISTRATION_KEYS}
    return (
        template_header()
        + "\n# --- Submission issue " + "-" * 57 + "\n"
        + _dump(issue_part)
        + "\n# --- Registration workflow " + "-" * 52 + "\n"
        + _dump(registration_part)
    )


def config_text(model: str) -> str:
    """Render the evaluation configuration: the template with `model: <model>`."""
    text = CONFIG_TEMPLATE.read_text()
    new_text, n = re.subn(r"^model:.*$", f"model: {model}", text, flags=re.MULTILINE)
    if n != 1:
        raise RegistrationError(f"No single 'model:' line in {CONFIG_TEMPLATE}")
    return new_text


# ---------------------------------------------------------------------------
# Issue comment
# ---------------------------------------------------------------------------


def comment_text(model: str, issue: int, blind: bool | None, repository: str) -> str:
    """
    Write the comment posted on the issue once the submission is registered.

    Parameters
    ----------
    model : str
        Emulator id (name of `data/<model>/` and of the branch `submission/<model>`).
    issue : int
        Number of the submission issue.
    blind : bool or None
        True for pre-release submissions, evaluated by the administrators.
    repository : str
        `<owner>/<repo>` of the benchmark repository.
    """
    branch = f"submission/{model}"
    intro = (
        f"Your submission has been approved and registered. The branch `{branch}`"
        f" now holds `data/{model}/metadata.yaml` (from this issue) and"
        f" `data/{model}/config.yaml` (evaluation settings). Please check the"
        " metadata; corrections can be made in the pull request.\n\n"
    )
    if blind:
        return intro + (
            "As the predictions were produced **before the release of the test"
            " dataset**, the benchmark administrators will evaluate them and open the"
            " pull request that adds the model to the benchmark. We will contact you"
            " in this issue to arrange the transfer of the predictions.\n"
        )
    return intro + f"""\
**Next steps**

1. Fork https://github.com/{repository} (if you have not done so yet) and get the
   branch of your submission:

   ```bash
   git clone https://github.com/<your-user>/ml-benchmark.git
   cd ml-benchmark
   git remote add upstream https://github.com/{repository}.git
   git fetch upstream {branch}
   git switch -c {branch} upstream/{branch}
   ```

2. Copy your predictions to `data/{model}/predictions/` (they are not versioned),
   then check and evaluate them:

   ```bash
   pixi run validate data/{model}/predictions
   pixi run evaluate -c data/{model}/config.yaml
   ```

3. Commit the evaluation (`metrics_summary.json` files) and push it to your fork:

   ```bash
   git add data/{model}/evaluation/*/*/metrics_summary.json
   git commit -m "Add evaluation of {model}"
   git push origin {branch}
   ```

4. Open a pull request from `{branch}` of your fork into `main` of
   `{repository}`, and mention #{issue} in its description.

The pull request shows a preview of the models table and the scoreboards. Once it
is merged, your model appears on the benchmark website and this issue is closed.
"""


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (defaults from the workflow's environment)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--body-file", type=Path,
        help="File with the issue body (default: environment variable ISSUE_BODY).",
    )
    parser.add_argument(
        "--issue", type=int, default=os.environ.get("ISSUE_NUMBER"),
        help="Issue number (default: ISSUE_NUMBER).",
    )
    parser.add_argument(
        "--created-at", default=os.environ.get("ISSUE_CREATED_AT"),
        help="Date the issue was opened, ISO 8601 (default: ISSUE_CREATED_AT).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT_DIR / "data",
        help="Directory where data/<MODEL>/ is created (default: data/).",
    )
    parser.add_argument(
        "--comment-file", type=Path,
        help="Write the comment for the issue to this file.",
    )
    parser.add_argument(
        "--repository", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY),
        help=f"<owner>/<repo> used in the comment (default: GITHUB_REPOSITORY or"
        f" {DEFAULT_REPOSITORY}).",
    )
    args = parser.parse_args()
    if args.issue is None or args.created_at is None:
        parser.error("the issue number and creation date are required")
    return args


def main() -> None:
    args = parse_args()
    body = args.body_file.read_text() if args.body_file else os.environ.get("ISSUE_BODY")
    if not body:
        sys.exit("[register] ERROR | Empty issue body.")

    try:
        form_fields = load_form_fields()
        answers = parse_issue_body(body, form_fields)
        submitted = dt.datetime.fromisoformat(args.created_at.replace("Z", "+00:00"))
        metadata = build_metadata(
            answers,
            form_fields,
            issue=args.issue,
            submitted=submitted.date(),
            approved=dt.datetime.now(dt.timezone.utc).date(),
        )
        model = metadata["emulator_id"]
        model_dir = args.data_dir / model
        if model_dir.exists():
            raise RegistrationError(f"{model_dir} already exists.")
        config = config_text(model)
    except RegistrationError as exc:
        sys.exit(f"[register] ERROR | {exc}")

    model_dir.mkdir(parents=True)
    (model_dir / "metadata.yaml").write_text(metadata_text(metadata))
    (model_dir / "config.yaml").write_text(config)
    print(f"[register] INFO | Wrote {model_dir}/metadata.yaml and config.yaml")

    if args.comment_file:
        text = comment_text(model, args.issue, metadata["blind"], args.repository)
        args.comment_file.write_text(text)

    # outputs for the next workflow steps
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"emulator_id={model}\n")
            f.write(f"blind={str(metadata['blind']).lower()}\n")


if __name__ == "__main__":
    main()
