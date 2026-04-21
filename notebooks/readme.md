# Notebooks - Prototyping & Debugging

This directory contains Jupyter notebooks used for rapid prototyping, data exploration, and debugging before implementing the final logic into the production scripts. These notebooks provide an interactive environment to test hypotheses and visualize the behavior of the models.

## Core Notebooks

### Data Analysis & Visualization
* **`data_visualization.ipynb`**: Focused on the visualization of raw $U$ and $V$ current components. Use this to understand the spatial distribution and scales of the dataset.
* **`RBF_weight_visualization.ipynb`**: Specifically designed to visualize the RBF weights. It helps in understanding how the high-dimensional current field is represented in the weight space.

### Modeling & Prototyping
* **`arima.ipynb`**: Implementation of baseline statistical models (ARIMA). Used as a benchmark to compare the hybrid neural network approach against traditional time-series forecasting.
* **`rbf_implementation.ipynb`**: Demonstrates the RBF network training process. It shows how clusters/centroids are formed and how the RBF parameters are extracted from the physical data.

* **`rbf_lstm.ipynb`**: **The core prototype.** This notebook contains the original logic for the hybrid approach. It was later refactored into the production Python scripts in the `src/rbf_lstm` folder. It remains highly useful for step-by-step debugging.

### Specialized Debugging
* **`RBF_LSTM_per_cluster.ipynb`**: Created for targeted debugging. Instead of predicting for the entire Lat/Long grid, it focuses on a single point (closest to **Bistyna**). It compares:
    1. A standard LSTM predicting $U$ and $V$ directly.
    2. The hybrid RBF + LSTM approach.

---


## Usage
These notebooks are intended to be "clickable" and easy to run. If you are troubleshooting the RBF-LSTM logic, start with `rbf_lstm.ipynb` to see the intermediate cell outputs and weight transformations.