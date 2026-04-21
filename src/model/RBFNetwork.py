import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class RBFNetwork:
    def __init__(
        self,
        spred: float | None = None,
        n_clusters: int = 10000,
    ) -> None:
        self.spred: float = spred
        self.n_clusters: int = n_clusters
        # self.scaler: StandardScaler = StandardScaler()
        self.model: LinearRegression = LinearRegression()
        self.kmeans: KMeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        self.cluster_centers: np.ndarray | None = None  # Will be set after training
        np.random.seed(42)

    def _gaussian_rbf(self, x: np.ndarray, center: np.ndarray) -> float:
        return np.exp(-(np.linalg.norm(x - center) ** 2) / (2 * self.spred**2))

    def _compute_rbf_activations(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        G = np.zeros((X.shape[0], len(centers)))

        for i, x_i in enumerate(X):
            for j, c_j in enumerate(centers):
                G[i, j] = self._gaussian_rbf(x_i, c_j)
        return G

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        # self.scaler.fit(x)
        # x_scal = self.scaler.transform(x)

        self.kmeans.fit(x)  # x_scal
        self.cluster_centers = self.kmeans.cluster_centers_
        if self.spred is None:
            d_max = np.max(pdist(self.cluster_centers))
            self.spred = d_max / (np.sqrt(2 * len(self.cluster_centers)))
            print(f"Spread set to {self.spred:.2f}")

        G_train = self._compute_rbf_activations(x, self.cluster_centers)  # x_scal
        self.model.fit(G_train, y)

        y_pred = self.model.predict(G_train)
        # print("RMSE for training:", np.sqrt(mean_squared_error(y, y_pred)))
        # print("R² for training:", r2_score(y, y_pred))
        RMSE = np.sqrt(mean_squared_error(y, y_pred))
        R_sqer = r2_score(y, y_pred)
        return RMSE, R_sqer

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x_scal = self.scaler.transform(x)
        G_test = self._compute_rbf_activations(x, self.cluster_centers)  # x_scal
        y_pred = self.model.predict(G_test)

        return y_pred

    def train_for_fix_cluster_and_scaler(self, x: np.ndarray, y: np.ndarray) -> None:
        # x_scal = self.scaler.transform(x)

        G_train = self._compute_rbf_activations(x, self.cluster_centers)  # x_scal
        self.model.fit(G_train, y)

        y_pred = self.model.predict(G_train)
        # print("RMSE for training:", np.sqrt(mean_squared_error(y, y_pred)))
        # print("R² for training:", r2_score(y, y_pred))
        RMSE = np.sqrt(mean_squared_error(y, y_pred))
        R_sqer = r2_score(y, y_pred)
        return RMSE, R_sqer

    def plot_cluster(self, x: np.ndarray):
        # x_scal = self.scaler.transform(x)

        # Predict clusters
        clusters = self.kmeans.predict(x)  # x_scal

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot points colored by cluster
        scatter = plt.scatter(x[:, 1], x[:, 0], c=clusters, cmap="tab10", alpha=0.6, s=10)  # x_scal

        # Plot cluster centers
        centers = self.kmeans.cluster_centers_
        plt.scatter(
            centers[:, 1],
            centers[:, 0],
            c="black",
            marker="x",
            s=200,
            linewidth=2,
            label="Cluster Centers",
        )

        plt.colorbar(scatter, label="Cluster")
        plt.xlabel("Longitude (scaled)")
        plt.ylabel("Latitude (scaled)")
        plt.title(f"RBF Network Clusters (n={self.n_clusters})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_uo_vo_plot(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))

        x_scal = self.scaler.transform(x)

        plots = [
            {"name": "UO", "pred": True, "ax": axes[0, 0]},
            {"name": "UO", "pred": False, "ax": axes[0, 1]},
            {"name": "VO", "pred": True, "ax": axes[1, 0]},
            {"name": "VO", "pred": False, "ax": axes[1, 1]},
        ]

        for plot in plots:
            name = plot["name"]
            col_name = f"{name.lower()}_pred" if plot["pred"] else name.lower()
            ax = plot["ax"]

            if plot["pred"] and plot["name"] == "UO":
                data = y_pred[:, 0]
            elif plot["pred"] and plot["name"] == "VO":
                data = y_pred[:, 1]
            elif not plot["pred"] and plot["name"] == "UO":
                data = y[:, 0]
            elif not plot["pred"] and plot["name"] == "VO":
                data = y[:, 1]

            df_pred = pd.DataFrame(
                {
                    "longitude": x_scal[:, 0],
                    "latitude": x_scal[:, 1],
                    col_name: data,
                }
            ).set_index(["latitude", "longitude"])
            pred_vo = df_pred.to_xarray()[col_name].unstack()

            pred_vo.plot(ax=ax, x="longitude", y="latitude", cmap="RdBu_r")
            pred = plot["pred"]
            ax.set_title(f"{'Predicted' if pred else 'True'} {name}")

        fig.suptitle(f"Number of cluster = {self.n_clusters}", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
