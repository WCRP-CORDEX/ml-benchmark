# Training

This repository  contains an example of how to train a model on the CORDEX ML-Bench data. It uses [DeepESD](https://doi.org/10.5194/gmd-15-6747-2022), a simple convolutional neural network, implemented in PyTorch. The example shows the workflow only; it is not tuned and its results are not meant as a reference.

- [`training.ipynb`](https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/training/training.ipynb): step-by-step tutorial that trains DeepESD for maximum temperature in one domain (SA), from loading and standardising the data to evaluating and plotting the predictions.

- [`training.py`](https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/training/training.py): the same model as a script that loops over all variables (`tasmax`, `pr`), domains (ALPS, NZ, SA) and training experiments, and saves the trained weights to `training/models/`.

## Running the example

1. Download the benchmark data (see [`data_download.ipynb`](https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/data/data_download.ipynb); the code expects it in `data/Bench-data/`.

2. Run the example from the repository folder, using the project environment (`pixi install` creates it the first time):
   - **Notebook**: open it in JupyterLab, which starts in your browser:
     ```bash
     pixi run lab training/training.ipynb
     ```
     Alternatively, open it in your own editor (e.g. VS Code).
   - **Script**: run it from inside the `training` folder, as it uses paths relative to it:
     ```bash
     cd training && pixi run python training.py
     ```

Once your model is trained, see [Submit your model](https://casadoj.github.io/ml-benchmark/contributing/submission/) for the format of the predictions.
