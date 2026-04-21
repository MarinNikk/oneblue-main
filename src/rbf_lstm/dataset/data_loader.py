"""Data loading and preprocessing for ocean current data."""

from typing import Any

import pandas as pd
import xarray as xr

from ..utils import Config, load_ocean_data, save_dataframe


def get_test_train_val_ds_size(uo: xr.Dataset, vo: xr.Dataset) -> tuple[int, int, int]:
    """Get dataset split sizes (keeping original function name from notebook)."""
    total_days: int = uo.time.size
    total_days_vo: int = vo.time.size
    assert total_days_vo == total_days

    train_end: int = int(0.70 * total_days) - 1
    test_end: int = int(0.85 * total_days) - 1
    val_end: int = int(total_days - 1)
    print(f"{train_end=}, {test_end=}, {val_end=}")
    return train_end, test_end, val_end


def split_data(
    var: xr.Dataset, train_end: int, test_end: int, val_end: int
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Split xarray dataset into train/test/val (keeping original function from notebook)."""
    return (
        var.isel(time=slice(0, train_end)),
        var.isel(time=slice(train_end, test_end)),
        var.isel(time=slice(test_end, val_end)),
    )


def drop_na_and_merge(uo_sub: xr.Dataset, vo_sub: xr.Dataset) -> pd.DataFrame:
    """
    Drop NaN values and merge datasets.

    Args:
        uo_sub: U-component ocean current dataset
        vo_sub: V-component ocean current dataset

    Returns:
        Merged DataFrame with non-NaN values
    """
    vo_sub_df: pd.DataFrame = vo_sub[["longitude", "latitude", "vo_detided"]].to_dataframe()
    uo_sub_df: pd.DataFrame = uo_sub[["longitude", "latitude", "uo_detided"]].to_dataframe()

    mask: pd.Series = (~vo_sub_df["vo_detided"].isna()) | (~uo_sub_df["uo_detided"].isna())
    vo_sub_filter: pd.DataFrame = vo_sub_df[mask].reset_index()
    uo_sub_filter: pd.DataFrame = uo_sub_df[mask].reset_index()

    base_col: list[str] = ["longitude", "latitude", "time"]
    assert (uo_sub_filter[base_col].to_numpy() == vo_sub_filter[base_col].to_numpy()).all()

    return pd.merge(
        uo_sub_filter,
        vo_sub_filter,
        on=base_col,
        # suffixes=('_uo', '_vo')
    )


def create_dict_map(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_end: int,
    test_end: int,
    val_end: int,
) -> tuple[dict[int, Any], dict[int, Any], dict[int, Any]]:
    """
    Create time mappings for dataset splits.

    Args:
        train_df, test_df, val_df: DataFrames for each split
        train_end, test_end, val_end: Split size endpoints

    Returns:
        Tuple of (int_to_data_train, int_to_data_test, int_to_data_val) mappings
    """
    int_to_data_train: dict[int, Any] = train_df["time"].drop_duplicates().to_dict()
    assert len(int_to_data_train) == train_end
    assert 0 in int_to_data_train
    assert train_end - 1 in int_to_data_train

    int_to_data_test: dict[int, Any] = test_df["time"].drop_duplicates().to_dict()
    assert len(int_to_data_test) == test_end - train_end
    assert 0 in int_to_data_train
    assert (test_end - train_end) - 1 in int_to_data_test

    int_to_data_val: dict[int, Any] = val_df["time"].drop_duplicates().to_dict()
    assert len(int_to_data_val) == val_end - test_end
    assert 0 in int_to_data_val
    assert (val_end - test_end) - 1 in int_to_data_val
    return int_to_data_train, int_to_data_test, int_to_data_val


def process_ocean_data(
    config: Config,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[int, Any],
    dict[int, Any],
    dict[int, Any],
]:
    """
    Load and process ocean current data into train/test/val splits.

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_df, test_df, val_df, int_to_data_train, int_to_data_test, int_to_data_val)
    """
    print("Loading ocean current datasets...")
    uo: xr.Dataset
    vo: xr.Dataset
    uo, vo = load_ocean_data(config.data.raw_data_dir, config.data.uo_file, config.data.vo_file)

    print("Calculating dataset split sizes...")
    train_end: int
    test_end: int
    val_end: int
    train_end, test_end, val_end = get_test_train_val_ds_size(uo, vo)

    print("Splitting datasets...")
    vo_train: xr.Dataset
    vo_test: xr.Dataset
    vo_val: xr.Dataset
    vo_train, vo_test, vo_val = split_data(vo, train_end, test_end, val_end)

    uo_train: xr.Dataset
    uo_test: xr.Dataset
    uo_val: xr.Dataset
    uo_train, uo_test, uo_val = split_data(uo, train_end, test_end, val_end)

    assert all(
        a.time.size == b.time.size
        for a, b in zip([uo_train, uo_test, uo_val], [vo_train, vo_test, vo_val], strict=True)
    )

    print("Processing and merging data splits...")
    train_df: pd.DataFrame = drop_na_and_merge(uo_train, vo_train)
    test_df: pd.DataFrame = drop_na_and_merge(uo_test, vo_test)
    val_df: pd.DataFrame = drop_na_and_merge(uo_val, vo_val)

    print("Saving processed DataFrames...")
    save_dataframe(train_df, f"{config.data.processed_data_dir}/train/dataframe.csv")
    save_dataframe(test_df, f"{config.data.processed_data_dir}/test/dataframe.csv")
    save_dataframe(val_df, f"{config.data.processed_data_dir}/val/dataframe.csv")

    print("Creating time mappings...")
    int_to_data_train: dict[int, Any]
    int_to_data_test: dict[int, Any]
    int_to_data_val: dict[int, Any]
    int_to_data_train, int_to_data_test, int_to_data_val = create_dict_map(
        train_df, test_df, val_df, train_end, test_end, val_end
    )

    print("Data processing complete:")
    print(f"  Train: {len(train_df):,} records, {len(int_to_data_train)} unique days")
    print(f"  Test:  {len(test_df):,} records, {len(int_to_data_test)} unique days")
    print(f"  Val:   {len(val_df):,} records, {len(int_to_data_val)} unique days")

    return train_df, test_df, val_df, int_to_data_train, int_to_data_test, int_to_data_val
