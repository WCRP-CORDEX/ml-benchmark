# Training

This folder contains illustrative scripts and notebooks to help you get started with training models for the CORDEX ML-Bench:

- **`training.ipynb`**: A notebook to visualize and learn how to train a model on the benchmark data.
- **`training.py`**: A simple script to train all models across all training experiments and domains.

## Evaluation Experiments

The following tables include a more detailed overview of the different evaluation experiments for the two available training experiments.

*Evaluation for the ESD pseudo-reality training experiment*

| Training Setup | Inference Set | Evaluation Type | Notes | Eval | Required |
|----------------|---------------|----------------|-------|------|----------|
| ESD “pseudo-reality”<br>Period: 1961–1980 | Historical (1981–2000) | Perfect Cross-Validation | Same GCM, perfectly | Error, Clim | X |
|  | Historical (1981–2000) | Imperfect Cross-Validation | Same GCM, imperfectly | Error, Clim | X |
|  | 2041–2060 + 2081–2100 | Perfect Extrapolation | Same GCM, perfectly | change signal | X |
|  | 2041–2060 + 2081–2100 | Imperfect Extrapolation| Same GCM, imperfectly | change signal | X |
|  | 2081–2100 | Perfect Extrapolation (Hard Transferability) | Different GCM, perfectly | change signal | X |


*Evaluation for the Emulator training experiment*

| Training Setup | Inference Set | Evaluation Type | Notes | Eval | Required |
|----------------|---------------|----------------|-------|------|----------|
| Emulator hist + future<br>Period: 1961–1980 + 2081–2100 | Historical (1981–2000) | Perfect Cross-Validation | Same GCM, perfectly | Error, Clim | X |
|  | Historical (1981–2000) | Imperfect Cross-Validation | Same GCM, imperfectly | Error, Clim | X |
|  | Historical (1981–2000) | Perfect Cross-Validation (Hard Transferability) | Different GCM, perfectly | Error, Clim | X |
|  | Historical (1981–2000) | Imperfect Cross-Validation (Hard Transferability) | Different GCM, imperfectly | Error, Clim | X |
|  | 2041–2060 + 2081–2100 | Perfect Interpolation | Same GCM, perfectly | change signal | X |
|  | 2041–2060 + 2081–2100 | Imperfect Interpolation | Same GCM, imperfectly | change signal | X |
|  | 2041–2060 + 2081–2100 | Perfect Interpolation (Hard Transferability) | Different GCM, perfectly | change signal | X |
|  | 2041–2060 + 2081–2100 | Imperfect Interpolation (Hard Transferability) | Different GCM, imperfectly | change signal | X |


## Baseline Models

CORDEX-ML-Bench includes a set of ML-based baseline models built on state-of-the-art developments. This allows users to compare the performance of their models against well-established baselines. The following table provides information about these models, along with links to repositories containing their implementations.

| Model       | Description | Reference | Implementation |
|-------------|-------------|-----------|----------------|
| DeepESD     | Convolutional neural network  | [Baño-Medina et al., 2024](https://gmd.copernicus.org/articles/15/6747/2022/) | [GitHub repository]() |

## Scoreboard

To track the performance of the different models compared for RCM emulation, we maintain a scoreboard with basic evaluation results for all models. The main requirement for inclusion in this table is that your model is associated with a scientific publication and has a publicly available code repository implementing it. For more information on adding your model, please contact contact.email@email.co.

| Model              | RMSE (°C)  | MAE (°C)  | R²    | Training Time     | Inference Speed (samples/sec) |
|--------------------|------------|-----------|-------|-------------------|-------------------------------|
| Model1             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model2             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model3             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model4             | XXX        | XXX       | XXX   | XXX               | XXX                           |
| Model5             | XXX        | XXX       | XXX   | XXX               | XXX                           |


















