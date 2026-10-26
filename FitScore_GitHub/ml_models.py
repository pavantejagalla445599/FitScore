"""
The ML core of the project — six algorithms across three explainable tasks:

1. K-Means clustering -> discovers behavioural "personas" from activity patterns
                         (unsupervised learning)
2. Four binary classifiers -> benchmark streak-risk prediction: Logistic Regression,
                              Decision Tree, Random Forest and Gaussian Naive Bayes
3. Linear Regression -> forecasts the user's next-day step count
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, roc_auc_score

PERSONA_NAMES = {0: "Under-active", 1: "Weekend Warrior", 2: "Consistent Mover", 3: "Elite Athlete"}


def build_user_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user - the behavioural fingerprint clustering runs on.

    avg_sleep is deliberately computed only over nights sleep was actually
    logged (df["sleep_logged"]), not the zero-filled TotalMinutesAsleep used
    for scoring. Averaging the zero-filled column would make a user's
    "average sleep" mostly measure how often they wear a device at night
    rather than how they actually sleep - confirmed on the real data here,
    where it silently pulled every persona's avg_sleep to well under half
    of a plausible 400+ min mean. sleep_log_rate is exposed alongside it so
    that distinction stays visible rather than hidden inside one number.
    """
    profile = df.groupby("Id").agg(
        avg_steps=("TotalSteps", "mean"),
        std_steps=("TotalSteps", "std"),
        avg_active_minutes=("VeryActiveMinutes", "mean"),
        avg_sedentary_minutes=("SedentaryMinutes", "mean"),
        sleep_log_rate=("sleep_logged", "mean"),
    ).fillna(0)

    logged_sleep = (
        df[df["sleep_logged"]].groupby("Id")["TotalMinutesAsleep"].mean()
    )
    profile["avg_sleep"] = profile.index.map(logged_sleep).fillna(0)
    return profile


def train_persona_clusters(profile_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42):
    features = profile_df[["avg_steps", "avg_active_minutes", "avg_sedentary_minutes", "avg_sleep"]]
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    raw_labels = model.fit_predict(X)

    profile_df = profile_df.copy()
    profile_df["cluster"] = raw_labels
    # re-rank clusters by avg_steps so persona names are consistent run to run,
    # since KMeans cluster numbering itself is arbitrary
    order = profile_df.groupby("cluster")["avg_steps"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    profile_df["cluster"] = profile_df["cluster"].map(remap)
    profile_df["persona"] = profile_df["cluster"].map(PERSONA_NAMES)
    return profile_df, model, scaler, remap


def predict_persona(avg_steps: float, avg_active_minutes: float, avg_sedentary_minutes: float,
                     avg_sleep: float, model: KMeans, scaler: StandardScaler, remap: dict) -> str:
    """
    Classifies a NEW (not-in-training) user against the already-fitted K-Means
    model - this is what "the model learned a pattern and can apply it to new
    input" actually means for clustering: scale the new point with the same
    scaler fit on the training population, ask the fitted model which centroid
    it's nearest to, then map that raw cluster id through the same rank-by-
    avg_steps remap used at training time so the persona name is consistent.
    """
    x = scaler.transform([[avg_steps, avg_active_minutes, avg_sedentary_minutes, avg_sleep]])
    raw_cluster = int(model.predict(x)[0])
    return PERSONA_NAMES[remap[raw_cluster]]


def build_streak_risk_dataset(df: pd.DataFrame):
    """
    Label: will tomorrow's steps fall below 70% of the trailing 3-day average?
    Built entirely from the activity table itself - no manual labelling needed.

    Feature set includes a 7-day rolling average and a 3-day sleep average
    alongside the original 1-day-lag / 3-day-rolling features - a longer
    window gives the model a less noisy read on a user's true baseline
    engagement level, which the 3-day-only version couldn't see past.
    """
    df = df.sort_values(["Id", "ActivityDate"]).copy()
    g = df.groupby("Id")
    df["steps_lag1"] = g["TotalSteps"].shift(1)
    df["steps_roll3"] = (
        g["TotalSteps"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["steps_roll7"] = (
        g["TotalSteps"].shift(1).rolling(7, min_periods=2).mean().reset_index(level=0, drop=True)
    )
    df["sleep_lag1"] = g["TotalMinutesAsleep"].shift(1)
    df["sleep_roll3"] = (
        g["TotalMinutesAsleep"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["day_of_week"] = df["ActivityDate"].dt.dayofweek
    df["next_day_steps"] = g["TotalSteps"].shift(-1)
    df["streak_risk_label"] = (df["next_day_steps"] < 0.7 * df["steps_roll3"]).astype(int)

    feature_cols = ["steps_lag1", "steps_roll3", "steps_roll7", "sleep_lag1", "sleep_roll3", "day_of_week"]
    model_df = df.dropna(subset=feature_cols + ["streak_risk_label"])
    return model_df, feature_cols


def train_streak_risk_models(model_df: pd.DataFrame, feature_cols: list):
    """
    Trains and benchmarks 4 classic, easy-to-explain classifiers on the same
    streak-risk task, so the project can talk about model comparison rather
    than a single black-box choice:

    - Logistic Regression : a weighted sum of features run through a sigmoid ->
                             probability of risk. Coefficients are directly readable.
    - Decision Tree        : a flowchart of yes/no splits ("steps < 4000? sleep < 6h?").
                             The most literally explainable model of the four.
    - Random Forest        : many decision trees voting together - usually more
                             accurate than one tree, still explainable via
                             feature importance.
    - Naive Bayes          : estimates risk from how likely each feature value is
                             under "will miss" vs "won't miss", assuming features
                             are independent. Fast, simple, a good baseline.

    Returns a dict of {name: (fitted_model, report_dict)} plus the name of
    the best performer by test-set accuracy.
    """
    X = model_df[feature_cols]
    y = model_df["streak_risk_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=42
        ),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        # ROC-AUC: a threshold-independent view of how well the model ranks
        # at-risk users above not-at-risk ones, using predicted probabilities
        # rather than the 0.5 cutoff classification_report uses. Reported
        # alongside accuracy/F1 because it's the standard way to judge an
        # imbalanced binary classifier without a single fixed threshold
        # baking in one particular trade-off.
        try:
            proba = model.predict_proba(X_test)[:, 1]
            report["roc_auc"] = float(roc_auc_score(y_test, proba))
        except (AttributeError, ValueError):
            report["roc_auc"] = None
        results[name] = (model, report)

    # Selection rule: F1-score on the at-risk (label=1) class, not raw
    # accuracy. The streak-risk label is imbalanced (roughly 1-in-9 user-days
    # are at-risk on this data), so a model that just predicts "not at risk"
    # almost every time scores a deceptively high accuracy while catching
    # almost none of the users it exists to catch (verified on this dataset:
    # the accuracy-best model had ~90% accuracy but ~5% recall on at-risk
    # cases). F1 balances precision and recall on the class that actually
    # drives the nudge/reward decision. Ties broken by accuracy.
    best_name = max(
        results,
        key=lambda n: (round(results[n][1]["1"]["f1-score"], 4), results[n][1]["accuracy"]),
    )
    return results, best_name


def build_steps_forecast_dataset(df: pd.DataFrame):
    """Regression target: how many steps will this user take tomorrow?
    Reuses the same lag features as the classifier, but predicts the raw number
    instead of a risk flag - a second, easy-to-explain ML task (Linear Regression)."""
    df = df.sort_values(["Id", "ActivityDate"]).copy()
    g = df.groupby("Id")
    df["steps_lag1"] = g["TotalSteps"].shift(1)
    df["steps_roll3"] = (
        g["TotalSteps"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["steps_roll7"] = (
        g["TotalSteps"].shift(1).rolling(7, min_periods=2).mean().reset_index(level=0, drop=True)
    )
    df["sleep_lag1"] = g["TotalMinutesAsleep"].shift(1)
    df["sleep_roll3"] = (
        g["TotalMinutesAsleep"].shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["day_of_week"] = df["ActivityDate"].dt.dayofweek
    df["next_day_steps"] = g["TotalSteps"].shift(-1)

    feature_cols = ["steps_lag1", "steps_roll3", "steps_roll7", "sleep_lag1", "sleep_roll3", "day_of_week"]
    model_df = df.dropna(subset=feature_cols + ["next_day_steps"])
    return model_df, feature_cols


def train_steps_forecast_model(model_df: pd.DataFrame, feature_cols: list):
    X = model_df[feature_cols]
    y = model_df["next_day_steps"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    metrics = {"mae": mean_absolute_error(y_test, preds), "r2": r2_score(y_test, preds)}
    return reg, metrics
