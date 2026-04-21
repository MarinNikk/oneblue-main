from dataclasses import dataclass


@dataclass
class EpochMetrics:
    loss: float = 0.0
    r2_lstm: float = 0.0
    r2_uv: float = 0.0
    r2_uv_true: float = 0.0

    def normalize(self, n: int):
        self.loss /= n
        self.r2_lstm /= n
        self.r2_uv /= n
        self.r2_uv_true /= n

    def __repr__(self):
        return (
            f"Metrics(loss={self.loss:.4f}, r2_lstm={self.r2_lstm:.4f}, r2_uv={self.r2_uv:.4f}, )"
            # f"r2_uv_true={self.r2_uv_true:.4f})"
        )
