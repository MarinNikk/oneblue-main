"""
Baseline models for ocean current prediction.

This module provides simple reference models that establish
minimum performance thresholds. Any sophisticated ML model
should significantly outperform these baselines.

Available baselines:
- SevenDayMeanBaseline: Predicts next day as 7-day average
"""

from .mean_baseline import SevenDayMeanBaseline, run_mean_baseline_experiment


__all__ = [
    "SevenDayMeanBaseline",
    "run_mean_baseline_experiment",
]
