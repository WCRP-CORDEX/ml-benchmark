# Scripts that build the website tables

These scripts generate the tables of the documentation site from the files in `models/<MODEL>/`. They are run automatically by GitHub  Actions; contributors do not need to run them.

| Script | Reads | Writes | Workflow |
| --- | --- | --- | --- |
| `scoreboard.py` | `models/<MODEL>/evaluation/<TRAIN_EXP>/<TEST_EXP>/metrics_summary.json`, `models/<MODEL>/metadata.yaml` | `docs/results/scoreboard/<train_exp>.md`, between `<!-- SCOREBOARD:START -->` and `<!-- SCOREBOARD:END -->` | `.github/workflows/scoreboard.yml` |
| `models_table.py` | `models/<MODEL>/metadata.yaml` | `docs/results/models.md`, between `<!-- MODELS:START -->` and `<!-- MODELS:END -->` | `.github/workflows/models.yml` |

On a pull request, the workflows only preview the tables in the job summary; on a push to `main` they regenerate the pages and commit them. Never edit the text between the markers by hand: it is overwritten on the next run.

To run them locally:

```bash
pixi run python gh_pages/scoreboard.py            # update the scoreboards in pages
pixi run python gh_pages/scoreboard.py --print    # print the tables instead
pixi run python gh_pages/models_table.py          # update the model table in pages
```