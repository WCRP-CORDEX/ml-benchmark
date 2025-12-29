# Submission

This folder contains illustrative scripts to help you generate submissions for the CORDEX ML-Bench:

- **`submission.py`**: This script demonstrates how to generate a submission for the benchmark. It runs predictions for all available test predictor files and organizes the outputs according to the required submission structure. It uses official templates from `format_predictions/templates/` to ensure correct spatial coordinates and variable attributes, and applies the required global metadata using `format_predictions/format.py`.

For additional details on submission formatting, refer to the [official documentation](https://docs.google.com/document/d/1zM6Aza1glzBJ8Ewvhu0_0vcm5dvctJ65Bypt1gnN2ck/edit?usp=sharing).
