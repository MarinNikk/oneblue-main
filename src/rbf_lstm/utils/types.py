import pandas as pd
from typing_extensions import TypedDict


class SplitData(TypedDict):
    time_to_coeff: dict
    time_to_metrics: dict
    int_to_data: dict
    rbf: object
    df: pd.DataFrame
