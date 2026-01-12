# Prediction Formatting and Templates

This folder provides utilities to prepare model predictions for CORDEX-ML-Benchmark:

- Use the provided NetCDF templates for each domain/variable to ensure the correct structure and coordinates.

## Contents

- `templates/`: Ready-to-fill templates (NetCDF) that match the benchmark's expected schema.

## Using the templates

Use the files in `./templates/` directly as a starting point. They contain the correct dimensions, coordinates, and a placeholder field for the requested variable. Replace values with your model predictions, preserving the structure.
