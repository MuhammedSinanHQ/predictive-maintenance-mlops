import pandas as pd
import numpy as np

# ------------------------
# Rolling Window Features
# ------------------------
def add_rolling_features(df, sensors=None, windows=[5, 10, 20]):
    """
    Adds rolling mean/std/min/max features for each sensor per engine (unit).
    """
    if sensors is None:
        sensors = [col for col in df.columns if col.startswith("sensor_")]

    df = df.sort_values(["unit", "time_cycle"]).copy()

    for sensor in sensors:
        for w in windows:
            df[f"{sensor}_r{w}_mean"] = (
                df.groupby("unit")[sensor].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            df[f"{sensor}_r{w}_std"] = (
                df.groupby("unit")[sensor].rolling(w, min_periods=1).std().reset_index(level=0, drop=True)
            )
            df[f"{sensor}_r{w}_min"] = (
                df.groupby("unit")[sensor].rolling(w, min_periods=1).min().reset_index(level=0, drop=True)
            )
            df[f"{sensor}_r{w}_max"] = (
                df.groupby("unit")[sensor].rolling(w, min_periods=1).max().reset_index(level=0, drop=True)
            )
    return df


# ------------------------
# Trend Features (Slope)
# ------------------------
def add_trend_features(df, sensors=None, window=20):
    """
    Computes slope (trend) over the last N cycles using a sliding window.
    """
    if sensors is None:
        sensors = [col for col in df.columns if col.startswith("sensor_")]

    df = df.sort_values(["unit", "time_cycle"]).copy()

    for sensor in sensors:
        trend_values = []

        for unit in df["unit"].unique():
            unit_slice = df[df["unit"] == unit].copy()
            vals = unit_slice[sensor].values
            slopes = []

            for i in range(len(vals)):
                if i < window:
                    slopes.append(0)  # not enough history
                else:
                    y = vals[i-window:i]
                    x = np.arange(window)
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)

            unit_slice[f"{sensor}_trend"] = slopes
            trend_values.append(unit_slice[f"{sensor}_trend"])

        df[f"{sensor}_trend"] = pd.concat(trend_values).sort_index()

    return df


# ------------------------
# Final Feature Generation
# ------------------------
def generate_features(df):
    sensors = [col for col in df.columns if col.startswith("sensor_")]

    df = add_rolling_features(df, sensors)
    df = add_trend_features(df, sensors)

    return df
