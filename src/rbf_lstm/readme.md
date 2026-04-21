# Ocean Current Prediction System: RBF-LSTM Hybrid Architecture

## 1. Project Overview
This project addresses the regression problem of predicting ocean current velocities (components $U$ and $V$) for the subsequent day ($t+1$). The system utilizes a novel two-stage architecture that combines **Radial Basis Function (RBF)** networks for spatial information compression and **Long Short-Term Memory (LSTM)** networks for temporal forecasting.

All implementation details, including the training scripts and model definitions for this hybrid approach, are located in the **`rbf_lstm`** subfolder.

## 2. Proposed Architecture

### Stage I: Information Compression & Feature Extraction (RBF)
Instead of feeding raw $UV$ data directly into the sequential model, the system first maps the input space using an RBF network.
* **Centroid Configuration:** The model currently utilizes **300 centroids** to approximate the $UV$ values for the current day ($t$). 
* **Dynamic Parameterization:** The number of centroids is a **configurable hyperparameter**, allowing for a trade-off between reconstruction fidelity and computational efficiency.
* **Function:** This stage acts as a "clever" feature extractor. It compresses complex spatial information into a set of RBF activation parameters (weights), effectively filtering noise and identifying latent spatial structures.



### Stage II: Temporal Sequence Prediction (LSTM)
The LSTM network operates in the transformed parameter space rather than the physical $UV$ space.
* **Input:** Historical sequences of RBF parameters (weights).
* **Target:** The predicted RBF parameters for the next day ($t+1$).
* **Rationale:** By learning the evolution of RBF weights, the LSTM captures the underlying dynamics of the ocean current structures rather than just the raw numerical fluctuations.



### Stage III: Physical Reconstruction
The final step involves a reconstruction phase where the predicted RBF parameters from the LSTM are passed back through the RBF basis functions to recover the predicted $U$ and $V$ velocity vectors.

---

## 3. Workflow Pipeline (`/rbf_lstm`)

| Step | Process | Output |
| :--- | :--- | :--- |
| 1 | **Data Input** | Raw spatial $U, V$ data at time $t$ |
| 2 | **Encoding** | Optimized RBF parameters/weights ($W_t$) |
| 3 | **Forecasting** | Predicted RBF parameters for $t+1$ ($\hat{W}_{t+1}$) |
| 4 | **Reconstruction** | Predicted physical currents $\hat{U}_{t+1}, \hat{V}_{t+1}$ |

---

## 4. Key Advantages
* **Efficient Representation:** Reducing the $UV$ field to 300 RBF weights significantly lowers the dimensionality of the sequence modeling task.
* **Noise Robustness:** The RBF pre-training phase serves as a spatial regularizer, preventing the LSTM from over-fitting to local turbulence.
* **Scalability:** Since the number of centroids is adjustable, the model can be easily scaled to higher resolutions or different geographic extents.


## 5. Getting Started

### Prerequisites
Ensure you have the Azure CLI and MLflow installed.

### Step 1: Download Experiments from Azure
To view the existing experiments, download the data from the Azure Storage Account:

```bash
az storage blob download-batch \
    --account-name oneblue \
    --source mlflow \
    --destination . \
    --auth-mode key
```

### Step 2: Launch MLflow UI
Run the following command to start the tracking server locally and explore the runs:

```bash
mlflow ui 
```

## 6. Experiment Analysis
In the MLflow dashboard, you will find multiple runs with various hyperparameters, following the naming convention:

```
rbf-lstm_c[centroids]_bs[batch_size]_h[hidden_size]_lr[learning_rate]
```

### Key Metrics Explained
We track two distinct types of $R^2$ scores to evaluate the performance of the hybrid architecture:
- **train_r2_lstm** / **val_r2_lstm**: These metrics measure how well the LSTM predicts the RBF weights for the next day.
- **train_r2_uv** / **val_r2_uv**: These metrics measure the accuracy of the final reconstructed $U$ and $V$ velocities compared to the ground truth.


### Current Findings & Challenges
As observed in the logged runs (e.g., val_r2_lstm ≈ 0.89 vs val_r2_uv ≈ 0.5):
 - **LSTM Performance**: The model performs exceptionally well at predicting the future weights of the RBF centroids. It captures the temporal dynamics of the compressed representation effectively.
 - **The Reconstruction Problem**: There is a significant performance drop after the reconstruction phase. While weight prediction is accurate, the final $R^2$ for $U$ and $V$ often drops to around 0.5.
 - **Sensitivity Issue**: This is the primary challenge of the current approach. Small variations or minor errors in the predicted RBF weights have a disproportionately large impact on the reconstructed physical velocities ($U$ and $V$). The system is highly sensitive to the precision of the RBF weight vector, making the mapping back to the physical domain the current bottleneck.


### 7. Run RBF LSTM
If you wish to run the RBF-LSTM model variant (located in the `rbf_lstm` module), follow these steps. 

**Important:** You must be at the **root level of the project hierarchy** to ensure all module paths are resolved correctly. Also, make sure to activate the project's **Conda environment** before execution:

```bash
# Activate the environment
conda activate oneblue

# Set MLflow tracking server URI (ensure server is running)
export MLFLOW_TRACKING_URI=[http://127.0.0.1:5000](http://127.0.0.1:5000)

# Run the RBF-LSTM training module
python3 -m rbf_lstm.main
```
