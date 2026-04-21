import numpy as np
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score

from .RBFNetwork import RBFNetwork


class RBFNetworkTrainer:
    def __init__(
        self,
        time: int = 0,
        spred: float = 0.5,
        n_clusters: int = 100,
        plot_cluster: bool = True,
        create_uo_vo_plot: bool = True,
    ) -> None:
        self.time: int = time
        self.spred: float = spred
        self.n_clusters: int = n_clusters
        self.plot_cluster: bool = plot_cluster
        self.create_uo_vo_plot: bool = create_uo_vo_plot
        self.rbf: RBFNetwork | None = None

    def drop_na(
        self, df: xr.Dataset, time: int = 0, name: str = "vo"
    ) -> tuple[np.ndarray, np.ndarray]:
        df = df.isel(time=time)
        col = f"{name}_detided"
        df_points = df[["longitude", "latitude", col]].to_dataframe()
        df_filter = df_points[~df_points[col].isna()].reset_index()  # use col here

        x = df_filter[["longitude", "latitude"]].to_numpy()
        y = df_filter[col].to_numpy()
        return x, y

    def load_and_prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        uo = xr.open_dataset("data/raw/adriatic_currents_uo_detided_2023_2024.nc")
        vo = xr.open_dataset("data/raw/adriatic_currents_vo_detided_2023_2024.nc")

        x_u, y_u = self.drop_na(uo, time=self.time, name="uo")
        x_v, y_v = self.drop_na(vo, time=self.time, name="vo")

        if not np.array_equal(x_u, x_v):
            raise ValueError("UO and VO coordinates do not match!")

        x = x_u
        y = np.stack((y_u, y_v), axis=1)

        return x, y

    def __call__(
        self,
    ):
        x, y = self.load_and_prepare_data()

        rbf = RBFNetwork(spred=self.spred, n_clusters=self.n_clusters)
        rbf.train(x, y)

        y_pred = rbf.predict(x)

        # TODO save to mplflow
        print("RMSE for testing:", np.sqrt(mean_squared_error(y, y_pred)))
        print("R² for testing:", r2_score(y, y_pred))

        if self.plot_cluster:
            rbf.plot_cluster(x)

        if self.create_uo_vo_plot:
            rbf.create_uo_vo_plot(x, y, y_pred)

        print("Coefficients shape:", rbf.model.coef_.shape)
        print("Intercept shape:", rbf.model.intercept_.shape)


# Example usage
if __name__ == "__main__":
    trainer = RBFNetworkTrainer(
        time=0, spred=None, n_clusters=100, plot_cluster=True, create_uo_vo_plot=True
    )

    trainer()
