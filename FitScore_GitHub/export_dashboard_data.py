"""
Exports everything the Streamlit app computes into one JSON payload, so a
static HTML preview of the dashboard can be built from REAL pipeline output
(not invented numbers) in an environment where Streamlit itself can't be
installed. This does not replace app.py — it's a read-only snapshot of what
app.py would show, for demo purposes only.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_pipeline import build_master_table, add_wellness_points, weekly_capped_points, assign_badges
from ml_models import (
    build_user_profile, train_persona_clusters,
    build_streak_risk_dataset, train_streak_risk_models,
    build_steps_forecast_dataset, train_steps_forecast_model,
)
from reward_engine import calculate_reward, tier_table
from wellness_tips import WELLNESS_TIPS
from sklearn.linear_model import LogisticRegression, LinearRegression


def j(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


df = build_master_table()
df = add_wellness_points(df)
df = assign_badges(df)
weekly = weekly_capped_points(df)

profile = build_user_profile(df)
profile, cluster_model, scaler, persona_remap = train_persona_clusters(profile)

risk_df, feature_cols = build_streak_risk_dataset(df)
results, best_name = train_streak_risk_models(risk_df, feature_cols)
risk_model, risk_report = results[best_name]

forecast_df, forecast_cols = build_steps_forecast_dataset(df)
forecast_model, forecast_metrics = train_steps_forecast_model(forecast_df, forecast_cols)

# ---- per-user daily wellness-score series (for the Overview line chart) ----
dates = sorted(df["ActivityDate"].dt.strftime("%Y-%m-%d").unique().tolist())
date_index = {d: i for i, d in enumerate(dates)}
users = sorted(df["Id"].unique().tolist())

series = {}
for uid in users:
    u = df[df["Id"] == uid].sort_values("ActivityDate")
    arr = [None] * len(dates)
    for d, s in zip(u["ActivityDate"].dt.strftime("%Y-%m-%d"), u["daily_wellness_score"]):
        arr[date_index[d]] = round(float(s), 1)
    series[str(uid)] = arr

# ---- latest-week leaderboard ----
latest = weekly[["iso_year", "iso_week"]].drop_duplicates().sort_values(["iso_year", "iso_week"]).iloc[-1]
board = weekly[(weekly["iso_year"] == latest["iso_year"]) & (weekly["iso_week"] == latest["iso_week"])] \
    .sort_values("weekly_points", ascending=False)
leaderboard = [{"Id": str(r.Id), "points": round(float(r.weekly_points), 1)} for r in board.itertuples()]

# ---- persona scatter ----
personas = [
    {"Id": str(uid), "persona": row["persona"], "avg_steps": round(float(row["avg_steps"])),
     "avg_active_minutes": round(float(row["avg_active_minutes"])), "avg_sleep": round(float(row["avg_sleep"]))}
    for uid, row in profile.iterrows()
]

# ---- classifier benchmark table ----
bench = []
for name, (_model, rep) in results.items():
    bench.append({
        "model": name, "accuracy": round(rep["accuracy"], 3),
        "precision": round(rep["1"]["precision"], 3), "recall": round(rep["1"]["recall"], 3),
        "f1": round(rep["1"]["f1-score"], 3), "roc_auc": round(rep["roc_auc"], 3) if rep.get("roc_auc") is not None else None,
    })
bench.sort(key=lambda r: -r["accuracy"])

# ---- exported model internals for live client-side inference (dashboard preview) ----
# The HTML preview has no Python backend, so "Predict New User" there needs the
# actual fitted model's parameters, not a re-trained approximation. Logistic
# Regression and Linear Regression are both a dot-product + (for LR) a sigmoid,
# so their exact math is cheap to reproduce in JS from coef_/intercept_ alone.
live_model = None
if isinstance(risk_model, LogisticRegression):
    live_model = {
        "type": "logreg", "features": feature_cols,
        "coef": risk_model.coef_[0].tolist(), "intercept": float(risk_model.intercept_[0]),
    }
# Linear Regression is always the forecast model (no benchmark needed there),
# so this export is unconditional.
live_forecast_model = {
    "type": "linreg", "features": forecast_cols,
    "coef": forecast_model.coef_.tolist(), "intercept": float(forecast_model.intercept_),
}
live_persona_model = {
    "features": ["avg_steps", "avg_active_minutes", "avg_sedentary_minutes", "avg_sleep"],
    "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
    "centroids": cluster_model.cluster_centers_.tolist(),
    "remap": {str(k): v for k, v in persona_remap.items()},
    "persona_names": {"0": "Under-active", "1": "Weekend Warrior", "2": "Consistent Mover", "3": "Elite Athlete"},
}

# ---- per-user: badges, wellness, persona, risk, forecast, reward ----
user_cards = []
for uid in users:
    u_df = df[df["Id"] == uid]
    persona = profile.loc[uid, "persona"]

    r_row = risk_df[risk_df["Id"] == uid].sort_values("ActivityDate").tail(1)
    risk_p = float(risk_model.predict_proba(r_row[feature_cols])[0][1]) if not r_row.empty else None

    f_row = forecast_df[forecast_df["Id"] == uid].sort_values("ActivityDate").tail(1)
    fc = float(forecast_model.predict(f_row[forecast_cols])[0]) if not f_row.empty else None

    reward = calculate_reward(u_df, persona=persona, risk_probability=risk_p, forecast_steps=fc)

    user_cards.append({
        "Id": str(uid),
        "persona": persona,
        "avg_wellness": round(float(u_df["daily_wellness_score"].mean()), 1),
        "badge_10k": int(u_df["badge_10k_steps"].sum()),
        "badge_comeback": int(u_df["badge_comeback_kid"].sum()),
        "risk_probability": round(risk_p, 3) if risk_p is not None else None,
        "forecast_steps": round(fc) if fc is not None else None,
        "reward_score": reward.reward_score,
        "reward_tier": reward.tier,
        "discount_pct": reward.discount_pct,
        "components": {
            "wellness": reward.wellness_component, "consistency": reward.consistency_component,
            "streak_safety": reward.risk_component, "forecast": reward.forecast_component,
        },
        "explanation": reward.explanation,
        "coaching_tip": reward.coaching_tip,
    })

payload = {
    "dates": dates,
    "users": [str(u) for u in users],
    "series": series,
    "leaderboard": leaderboard,
    "personas": personas,
    "classifier_benchmark": bench,
    "best_classifier": best_name,
    "forecast_metrics": {"mae": round(forecast_metrics["mae"], 1), "r2": round(forecast_metrics["r2"], 3)},
    "reward_policy": tier_table().to_dict(orient="records"),
    "user_cards": user_cards,
    "wellness_tips": WELLNESS_TIPS,
    "n_users": len(users),
    "n_days": len(dates),
    "date_range": f"{dates[0]} to {dates[-1]}",
    "live_risk_model": live_model,
    "live_forecast_model": live_forecast_model,
    "live_persona_model": live_persona_model,
    "feature_cols": feature_cols,
    "forecast_cols": forecast_cols,
}

with open("dashboard_export.json", "w") as f:
    json.dump(payload, f, default=j)

print(f"Exported {len(users)} users x {len(dates)} days to dashboard_export.json")
print(f"Best classifier: {best_name}  |  Forecast MAE={forecast_metrics['mae']:.0f}, R2={forecast_metrics['r2']:.2f}")
print(f"Reward tiers: {pd.Series([c['reward_tier'] for c in user_cards]).value_counts().to_dict()}")
