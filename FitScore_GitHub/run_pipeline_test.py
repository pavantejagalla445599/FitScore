"""
Runs the exact same pipeline app.py wires into the dashboard, headless.
This is how the project was validated in a sandbox where streamlit/plotly
aren't installable (no internet access) — every function app.py calls is
exercised here directly, so a clean run here means app.py's data/ML layer
is correct; only the Streamlit rendering itself is untested by this script.
"""
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

Path("fitscore.db").unlink(missing_ok=True)  # fresh DB each test run

import numpy as np
import pandas as pd

from data_pipeline import build_master_table, add_wellness_points, weekly_capped_points, assign_badges
from ml_models import (
    build_user_profile, train_persona_clusters,
    build_streak_risk_dataset, train_streak_risk_models,
    build_steps_forecast_dataset, train_steps_forecast_model,
)
from nlp_sentiment import score_sentiment
from rag_coach import build_retriever, ask_fitcoach
import sql_store
from reward_engine import calculate_reward, tier_table, REWARD_TIERS

print("=" * 70)
print("1. DATA PIPELINE")
print("=" * 70)
df = build_master_table()
print(f"master table: {df.shape[0]:,} rows, {df['Id'].nunique()} users, "
      f"{df['ActivityDate'].min().date()} to {df['ActivityDate'].max().date()}")
print("nulls after join:\n", df.isna().sum()[lambda s: s > 0].to_string() or "  none")
assert df["TotalMinutesAsleep"].isna().sum() == 0, "sleep fillna(0) should leave no NaNs"

df = add_wellness_points(df)
print(f"\ndaily_wellness_score  min={df['daily_wellness_score'].min():.1f} "
      f"max={df['daily_wellness_score'].max():.1f} mean={df['daily_wellness_score'].mean():.1f} "
      f"(scale is 0-90: 40 steps + 30 active + 20 sleep)")
assert df["daily_wellness_score"].max() <= 90.01, "wellness score exceeded documented 0-90 scale"
assert df["daily_wellness_score"].min() >= 0, "wellness score went negative"

df = assign_badges(df)
print(f"badge_10k_steps days: {int(df['badge_10k_steps'].sum())} "
      f"({100*df['badge_10k_steps'].mean():.1f}% of user-days)")
print(f"badge_comeback_kid days: {int(df['badge_comeback_kid'].sum())}")

weekly = weekly_capped_points(df)
print(f"weekly leaderboard rows: {len(weekly)}, weekly_points capped at 500 "
      f"(max seen: {weekly['weekly_points'].max():.0f})")

print("\n" + "=" * 70)
print("2. K-MEANS PERSONA CLUSTERING")
print("=" * 70)
profile = build_user_profile(df)
profile, cluster_model, scaler, persona_remap = train_persona_clusters(profile)
print(profile.groupby("persona").agg(
    users=("cluster", "size"), avg_steps=("avg_steps", "mean"),
    avg_sleep=("avg_sleep", "mean")
).round(0).to_string())
n_personas = profile["persona"].nunique()
assert n_personas == 4, f"expected 4 personas, got {n_personas}"

print("\n" + "=" * 70)
print("3. STREAK-RISK CLASSIFICATION (4-way benchmark)")
print("=" * 70)
risk_df, feature_cols = build_streak_risk_dataset(df)
print(f"modelling rows: {len(risk_df):,} (after dropping rows without full lag history)")
print(f"label balance: at-risk={risk_df['streak_risk_label'].mean()*100:.1f}%  "
      f"not-at-risk={100 - risk_df['streak_risk_label'].mean()*100:.1f}%")
# leakage guard: features must not include same-day or future information
leak_check = [c for c in feature_cols if c in ("TotalSteps", "next_day_steps")]
assert not leak_check, f"potential leakage: {leak_check} in feature_cols"

results, best_name = train_streak_risk_models(risk_df, feature_cols)
rows = []
for name, (model, rep) in results.items():
    rows.append({
        "model": name, "accuracy": rep["accuracy"],
        "precision_at_risk": rep["1"]["precision"], "recall_at_risk": rep["1"]["recall"],
        "f1_at_risk": rep["1"]["f1-score"], "roc_auc": rep.get("roc_auc"), "support_at_risk": rep["1"]["support"],
    })
bench = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
print(bench.round(3).to_string(index=False))
print(f"\nbest model (by F1 on at-risk class, tie-broken by accuracy): {best_name}")

print("\n" + "=" * 70)
print("4. LINEAR REGRESSION — NEXT-DAY STEP FORECAST")
print("=" * 70)
forecast_df, forecast_cols = build_steps_forecast_dataset(df)
forecast_model, forecast_metrics = train_steps_forecast_model(forecast_df, forecast_cols)
print(f"MAE = {forecast_metrics['mae']:.0f} steps, R^2 = {forecast_metrics['r2']:.3f}")
naive_mae = float((forecast_df["next_day_steps"] - forecast_df["steps_roll3"]).abs().mean())
print(f"(naive baseline — predicting yesterday's 3-day rolling average — MAE = {naive_mae:.0f} steps, "
      f"for comparison)")

print("\n" + "=" * 70)
print("5. SQL LAYER (SQLite)")
print("=" * 70)
sql_store.build_database(df, weekly)
sql_store.save_reflection("TESTUSER", "2025-W05", "Felt great and energised this week!", "Positive", 0.15)
hist = sql_store.get_user_reflections("TESTUSER")
top5 = sql_store.get_top_users(limit=5)
print("get_top_users(5):\n", top5.to_string(index=False))
assert len(hist) == 1, "reflection round-trip through SQLite failed"
print("reflection round-trip through SQLite: OK")

print("\n" + "=" * 70)
print("6. NLP SENTIMENT")
print("=" * 70)
for txt in ["Great week, felt strong and motivated every day!",
            "Exhausted and struggling, missed most workouts.",
            "An average week, nothing special either way."]:
    r = score_sentiment(txt)
    print(f"  [{r['label']:8s}] score={r['score']:+.3f}  \"{txt}\"")

print("\n" + "=" * 70)
print("7. RAG — ASK FITCOACH (TF-IDF retrieval)")
print("=" * 70)
retriever = build_retriever(k=2)
for q in ["How do I improve my sleep score?", "I keep missing my step goal", "How does the leaderboard work?"]:
    tips = ask_fitcoach(q, retriever)
    print(f"  Q: {q}")
    for t in tips:
        print(f"     -> [{t['topic']}] {t['text'][:80]}...")

print("\n" + "=" * 70)
print("8. REWARD ENGINE — full population, then one example user")
print("=" * 70)
all_scores = []
for uid in df["Id"].unique():
    u_df = df[df["Id"] == uid]
    persona = profile.loc[uid, "persona"]
    r_row = risk_df[risk_df["Id"] == uid].sort_values("ActivityDate").tail(1)
    risk_p = float(results[best_name][0].predict_proba(r_row[feature_cols])[0][1]) if not r_row.empty else None
    f_row = forecast_df[forecast_df["Id"] == uid].sort_values("ActivityDate").tail(1)
    fc = float(forecast_model.predict(f_row[forecast_cols])[0]) if not f_row.empty else None
    reward = calculate_reward(u_df, persona=persona, risk_probability=risk_p, forecast_steps=fc)
    all_scores.append({"Id": uid, "reward_score": reward.reward_score, "tier": reward.tier})

reward_pop = pd.DataFrame(all_scores)
print("reward_score distribution across all users:")
print(reward_pop["reward_score"].describe().round(1).to_string())
print("\ntier distribution with CURRENT thresholds (Platinum>=79, Gold>=73, Silver>=66, Bronze<66):")
print(reward_pop["tier"].value_counts().reindex(["Platinum", "Gold", "Silver", "Bronze"]).fillna(0).astype(int).to_string())

print("\nquantile-based calibration check (25/50/75/90th percentiles of reward_score):")
print(reward_pop["reward_score"].quantile([0.25, 0.5, 0.75, 0.90]).round(1).to_string())

# ---- one concrete example user, full walkthrough ----
example_uid = reward_pop.sort_values("reward_score", ascending=False).iloc[len(reward_pop)//4]["Id"]
u_df = df[df["Id"] == example_uid]
persona = profile.loc[example_uid, "persona"]
r_row = risk_df[risk_df["Id"] == example_uid].sort_values("ActivityDate").tail(1)
risk_p = float(results[best_name][0].predict_proba(r_row[feature_cols])[0][1])
f_row = forecast_df[forecast_df["Id"] == example_uid].sort_values("ActivityDate").tail(1)
fc = float(forecast_model.predict(f_row[forecast_cols])[0])
reward = calculate_reward(u_df, persona=persona, risk_probability=risk_p, forecast_steps=fc)

example = {
    "user_id": str(example_uid),
    "avg_wellness_score": round(float(u_df["daily_wellness_score"].mean()), 1),
    "persona": persona,
    "streak_risk_probability": round(risk_p, 3),
    "forecast_next_day_steps": round(fc),
    "reward_score": reward.reward_score,
    "reward_tier": reward.tier,
    "illustrative_discount_pct": reward.discount_pct,
    "components": {
        "wellness": reward.wellness_component, "consistency": reward.consistency_component,
        "streak_safety": reward.risk_component, "forecast": reward.forecast_component,
    },
    "explanation": reward.explanation,
}
print("\nEXAMPLE CUSTOMER:")
print(json.dumps(example, indent=2, default=str))

with open("pipeline_results.json", "w") as f:
    json.dump({
        "n_users": int(df["Id"].nunique()), "n_rows": int(len(df)),
        "classifier_benchmark": bench.round(4).to_dict(orient="records"),
        "best_classifier": best_name,
        "forecast_mae": round(forecast_metrics["mae"], 1), "forecast_r2": round(forecast_metrics["r2"], 3),
        "n_personas": int(n_personas),
        "reward_score_distribution": reward_pop["reward_score"].describe().round(1).to_dict(),
        "reward_tier_distribution": reward_pop["tier"].value_counts().to_dict(),
        "example_customer": example,
    }, f, indent=2, default=str)

print("\n" + "=" * 70)
print("ALL STAGES COMPLETED WITHOUT ERROR")
print("=" * 70)
