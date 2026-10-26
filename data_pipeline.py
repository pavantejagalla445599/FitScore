"""
Data loading + feature engineering for the FitScore wellness-rewards project.

Dataset: "FitBit Fitness Tracker Data" (Kaggle, arashnic/fitbit) - CC0 public domain.
https://www.kaggle.com/datasets/arashnic/fitbit
Download it, unzip, and place these two files in a local `data/` folder:
    - dailyActivity_merged.csv
    - sleepDay_merged.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")


def load_daily_activity(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "dailyActivity_merged.csv")
    df["ActivityDate"] = pd.to_datetime(df["ActivityDate"])
    return df


def load_sleep(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "sleepDay_merged.csv")
    df["SleepDay"] = pd.to_datetime(df["SleepDay"], format="mixed").dt.normalize()
    df = df.rename(columns={"SleepDay": "ActivityDate"})
    # a handful of users log sleep more than once a day - keep the longest record
    df = df.sort_values("TotalMinutesAsleep", ascending=False).drop_duplicates(["Id", "ActivityDate"])
    return df[["Id", "ActivityDate", "TotalMinutesAsleep", "TotalTimeInBed"]]


def build_master_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per user per day: activity + sleep joined."""
    activity = load_daily_activity(data_dir)
    sleep = load_sleep(data_dir)
    df = activity.merge(sleep, on=["Id", "ActivityDate"], how="left")
    # Keep a flag for whether sleep was actually logged BEFORE zero-filling.
    # The zero-fill below is a deliberate scoring rule (no log = no sleep
    # points), but ml_models.py needs this flag so persona clustering
    # averages real sleep nights instead of being dragged down by zeros
    # on nights the device just wasn't worn.
    df["sleep_logged"] = df["TotalMinutesAsleep"].notna()
    df["TotalMinutesAsleep"] = df["TotalMinutesAsleep"].fillna(0)
    return df.sort_values(["Id", "ActivityDate"]).reset_index(drop=True)


def add_wellness_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based point mechanic from the project brief:
    steps + active minutes (cardio weighted higher) + sleep quality,
    each individually capped so no single lever can be over-farmed.
    """
    df = df.copy()
    df["steps_points"] = (df["TotalSteps"] / 10_000 * 40).clip(upper=40)
    df["active_points"] = (
        df["VeryActiveMinutes"] * 1.0
        + df["FairlyActiveMinutes"] * 0.6
        + df["LightlyActiveMinutes"] * 0.2
    ).clip(upper=30)
    ideal_sleep = df["TotalMinutesAsleep"].between(420, 540)  # 7-9 hours
    sleep_ratio = (df["TotalMinutesAsleep"] / 480).clip(upper=1.1)
    df["sleep_points"] = np.where(ideal_sleep, 20, sleep_ratio * 15)
    df["sleep_points"] = df["sleep_points"].clip(0, 20)
    df["daily_wellness_score"] = (
        df["steps_points"] + df["active_points"] + df["sleep_points"]
    ).round(1)
    return df


def weekly_capped_points(df: pd.DataFrame, weekly_cap: int = 500) -> pd.DataFrame:
    """Sums daily points per user per ISO week, capped so a single great day
    can't be farmed into an unlimited weekly total (resets leaderboard weekly)."""
    df = df.copy()
    iso = df["ActivityDate"].dt.isocalendar()
    df["iso_year"], df["iso_week"] = iso["year"], iso["week"]
    weekly = (
        df.groupby(["Id", "iso_year", "iso_week"])["daily_wellness_score"]
        .sum()
        .clip(upper=weekly_cap)
        .reset_index(name="weekly_points")
    )
    return weekly


def assign_badges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["badge_10k_steps"] = df["TotalSteps"] >= 10_000
    df["badge_early_bird"] = df["VeryActiveMinutes"] >= 20
    df["rolling_avg_7d"] = df.groupby("Id")["TotalSteps"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["badge_comeback_kid"] = (df["TotalSteps"] > df["rolling_avg_7d"] * 1.3) & (
        df["rolling_avg_7d"] < 5_000
    )
    return df


def score_single_day(steps: float, very_active: float, fairly_active: float,
                      lightly_active: float, sleep_minutes: float) -> float:
    """
    Wellness score for one hypothetical day of inputs — used by the "Predict
    New User" feature. Deliberately reuses add_wellness_points() itself on a
    one-row frame rather than re-deriving the formula, so there is exactly
    one place the 40/30/20-point scoring rule is implemented; the existing-
    user scoring path and the new-user prediction path can never drift apart.
    """
    row = pd.DataFrame([{
        "TotalSteps": steps, "VeryActiveMinutes": very_active,
        "FairlyActiveMinutes": fairly_active, "LightlyActiveMinutes": lightly_active,
        "TotalMinutesAsleep": sleep_minutes,
    }])
    return float(add_wellness_points(row)["daily_wellness_score"].iloc[0])
