"""
Feature engineering for the early-warning model.

For each machine's time series we build rolling-window features (mean, std,
min/max, rate-of-change) over multiple window sizes, then label each row
with `label` = 1 if a failure occurs within the next HORIZON steps (the
early-warning target), 0 otherwise. Rows after a failure are dropped.
"""
import numpy as np
import pandas as pd

SENSORS = ["vibration", "temperature", "current", "pressure"]
WINDOWS = [6, 24, 72]          # short / medium / long rolling windows (hours)
HORIZON = 48                   # predict failure within the next 48 hours


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "step"]).reset_index(drop=True)
    out = [df]
    g = df.groupby("machine_id")
    for w in WINDOWS:
        for s in SENSORS:
            roll = g[s].rolling(w, min_periods=max(2, w // 3))
            out.append(roll.mean().reset_index(drop=True).rename(f"{s}_mean_{w}"))
            out.append(roll.std().reset_index(drop=True).rename(f"{s}_std_{w}"))
            out.append(roll.max().reset_index(drop=True).rename(f"{s}_max_{w}"))
            out.append(roll.min().reset_index(drop=True).rename(f"{s}_min_{w}"))
    # rate of change (short vs longer window) + first difference
    for s in SENSORS:
        out.append(g[s].diff().reset_index(drop=True).rename(f"{s}_diff1"))
        out.append((g[s].diff(6)).reset_index(drop=True).rename(f"{s}_diff6"))
    feat = pd.concat(out, axis=1)
    return feat


def add_label(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """label=1 if a failure_event occurs within `horizon` steps ahead, same machine."""
    df = df.copy()
    df["label"] = 0
    for mid, sub in df.groupby("machine_id"):
        fail_idx = sub.index[sub["failure_event"] == 1]
        if len(fail_idx) == 0:
            continue
            # (dataset has at most one failure per machine by construction)
        f = fail_idx[0]
        f_step = df.loc[f, "step"]
        window_mask = (df["machine_id"] == mid) & (df["step"] >= f_step - horizon) & (df["step"] < f_step)
        df.loc[window_mask, "label"] = 1
    return df


def build_feature_table(raw_csv="sensor_data.csv", out_csv="features.csv"):
    import os
    df = pd.read_csv(raw_csv, parse_dates=["timestamp"])
    df = add_label(df)
    feat = add_rolling_features(df)
    feat = feat.dropna().reset_index(drop=True)
    feat.to_csv(out_csv, index=False, encoding="utf-8")
    os.makedirs("data", exist_ok=True)
    feat.to_csv(os.path.join("data", out_csv), index=False, encoding="utf-8")
    print(f"feature table: {feat.shape[0]} rows x {feat.shape[1]} cols, "
          f"positive rate: {feat['label'].mean():.3f}")
    return feat


if __name__ == "__main__":
    build_feature_table()
