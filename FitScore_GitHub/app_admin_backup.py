import pandas as pd
import streamlit as st
import plotly.express as px

from data_pipeline import (
    build_master_table,
    add_wellness_points,
    weekly_capped_points,
    assign_badges,
    score_single_day,
)
from ml_models import (
    build_user_profile,
    train_persona_clusters,
    predict_persona,
    build_streak_risk_dataset,
    train_streak_risk_models,
    build_steps_forecast_dataset,
    train_steps_forecast_model,
)
from nlp_sentiment import score_sentiment
from rag_coach import build_retriever, ask_fitcoach
import sql_store
from reward_engine import calculate_reward, calculate_reward_from_inputs, tier_table

st.set_page_config(page_title="FitScore — Wellness Rewards", page_icon="🏃", layout="wide")

# ---------------------------------------------------------------------------
# Light visual polish. Streamlit's own widgets can't be restyled much beyond
# this without custom components, so this focuses on the things that make
# the biggest visible difference: a real header band, card-style metrics,
# and a consistent accent color instead of Streamlit's stock look.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
  --fs-accent: #e0592a;
  --fs-accent-2: #f0803f;
  --fs-accent-soft: #fdece3;
  --fs-ink: #1c2333;
  --fs-muted: #5c6470;
  --fs-line: #e6e9ef;
  --fs-panel: #ffffff;
  --fs-bg: #f4f6fa;
}
.stApp { background: var(--fs-bg); }
section.main > div { padding-top: 8px; }
html, body, [class*="css"] { color: var(--fs-ink); }
.fs-hero {
  background: linear-gradient(135deg, #10142a 0%, #1e2a52 55%, #2a3a6b 100%);
  padding: 30px 34px;
  border-radius: 16px;
  margin-bottom: 22px;
  color: #fff;
  box-shadow: 0 8px 24px rgba(20,30,70,0.18);
}
.fs-hero h1 { margin: 0 0 8px 0; font-size: 30px; letter-spacing: -0.3px; }
.fs-hero p { margin: 0; opacity: 0.88; font-size: 14.5px; line-height: 1.55; }
div[data-testid="stMetric"] {
  background: var(--fs-panel);
  border: 1px solid var(--fs-line);
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: 0 2px 6px rgba(20,30,70,0.05);
  transition: transform .12s ease, box-shadow .12s ease;
}
div[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(20,30,70,0.10);
}
div[data-testid="stMetric"] * { color: var(--fs-ink) !important; }
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
  font-size: 12.5px; color: var(--fs-muted) !important; font-weight: 600;
}
div[data-testid="stMetricValue"] { color: var(--fs-ink) !important; font-weight: 800; font-size: 26px; }
button[data-baseweb="tab"] { font-size: 14px; font-weight: 600; color: var(--fs-muted); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--fs-accent) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--fs-accent) !important; height: 3px; }
div[data-baseweb="tab-border"] { background-color: var(--fs-line); }
h1, h2, h3 { color: var(--fs-ink); }
.stApp h3 { font-weight: 700; }
div[data-testid="stDataFrame"] {
  border: 1px solid var(--fs-line);
  border-radius: 10px;
  overflow: hidden;
}
.stButton > button, .stFormSubmitButton > button {
  background: var(--fs-accent);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  padding: 8px 20px;
  transition: background .12s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  background: var(--fs-accent-2); color: #fff;
}
div[data-baseweb="input"] input { border-radius: 8px !important; }
section[data-testid="stSidebar"] { background: var(--fs-panel); border-right: 1px solid var(--fs-line); }
.fs-rule-box {
  background: #f8f9fc;
  border-left: 3px solid var(--fs-accent);
  border-radius: 8px;
  padding: 14px 18px;
  margin-bottom: 12px;
  font-size: 14px;
  line-height: 1.9;
  color: var(--fs-ink);
}
.fs-rule-box b { color: #000; }
.fs-assumption {
  background: #fff8e8;
  border-left: 3px solid #b8860b;
  border-radius: 8px;
  padding: 11px 16px;
  margin-bottom: 10px;
  font-size: 13.5px;
  color: #6b5109 !important;
}
div[data-testid="stAlert"] { border-radius: 10px; }
.fs-flow {
  display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap;
  margin: 8px 0 20px 0;
}
.fs-flow-step {
  flex: 1; min-width: 150px;
  background: linear-gradient(135deg, #fff 0%, #f8f9fc 100%);
  border: 1px solid var(--fs-line); border-top: 3px solid var(--fs-accent);
  border-radius: 10px; padding: 14px 16px; font-size: 13.5px; line-height: 1.5;
  box-shadow: 0 2px 6px rgba(20,30,70,0.05);
}
.fs-flow-step b { color: var(--fs-accent); font-size: 14px; }
.fs-flow-arrow {
  display: flex; align-items: center; font-size: 22px; color: var(--fs-accent);
  font-weight: 700;
}
@media (max-width: 900px){ .fs-flow-arrow{ display:none; } }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = build_master_table()
    df = add_wellness_points(df)
    df = assign_badges(df)
    return df


@st.cache_data
def run_ml(df):
    profile = build_user_profile(df)
    profile, cluster_model, scaler, persona_remap = train_persona_clusters(profile)

    risk_df, feature_cols = build_streak_risk_dataset(df)
    results, best_name = train_streak_risk_models(risk_df, feature_cols)
    best_model, best_report = results[best_name]

    forecast_df, forecast_cols = build_steps_forecast_dataset(df)
    forecast_model, forecast_metrics = train_steps_forecast_model(forecast_df, forecast_cols)

    return (
        profile, cluster_model, scaler, persona_remap,
        risk_df, feature_cols, results, best_name, best_model, best_report,
        forecast_df, forecast_cols, forecast_model, forecast_metrics,
    )


st.markdown("""
<div class="fs-hero">
  <h1>🏃 FitScore</h1>
  <p style="font-size:16px; opacity:0.95; margin-bottom:10px;">
  Turning wearable data into fair, personalised insurance rewards.</p>
  <p style="font-size:13px; opacity:0.75;">
  Wearable activity &amp; sleep → wellness score → ML behavioural intelligence →
  transparent reward tier → illustrative premium discount</p>
</div>
""", unsafe_allow_html=True)

df = load_data()
(
    profile, cluster_model, scaler, persona_remap,
    risk_df, feature_cols, model_results, best_name, risk_model, report,
    forecast_df, forecast_cols, forecast_model, forecast_metrics,
) = run_ml(df)

# --- SQL layer: everything computed above gets written to fitscore.db ---
weekly_all = weekly_capped_points(df)
sql_store.build_database(df, weekly_all)

# --- RAG layer: LangChain TF-IDF retriever over the wellness_tips knowledge base ---
@st.cache_resource
def get_retriever():
    return build_retriever(k=3)

retriever = get_retriever()

users = sorted(df["Id"].unique())
st.sidebar.markdown("### 👤 Customer")
st.sidebar.caption("Pick a customer to see their full profile across every tab.")
selected_user = st.sidebar.selectbox("Select customer", users, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("### 🧠 ML engine status")
st.sidebar.metric(
    f"Best classifier: {best_name}", f"F1={report['1']['f1-score']:.3f}",
    help="Selected by F1-score on the at-risk class (not raw accuracy) — see the Streak Risk tab for why."
)
if report.get("roc_auc") is not None:
    st.sidebar.caption(f"ROC-AUC: {report['roc_auc']:.3f} · Accuracy: {report['accuracy']*100:.0f}%")
st.sidebar.caption(f"Trained on {df['Id'].nunique()} customers × {df['ActivityDate'].nunique()} days.")

tab_home, tab_rewards, tab_predict, tab_overview, tab_persona, tab_risk, tab_forecast, tab_rules, tab_leaderboard, tab_reflect, tab_coach = st.tabs(
    ["🏠 Home", "💳 Insurance Rewards", "🧮 Predict New Customer", "📊 Customer Activity",
     "🧬 Persona", "⚠️ Streak Risk", "🔮 Forecast", "📐 Business Rules",
     "🏆 Leaderboard", "📝 Reflection", "🤖 Ask FitCoach"]
)

user_df = df[df["Id"] == selected_user].sort_values("ActivityDate")

with tab_home:
    st.markdown("## The problem")
    st.markdown(
        "Insurers increasingly want to **reward customers for healthy behaviour** with premium "
        "discounts — it lowers their long-term risk and keeps customers engaged. But they need a "
        "**fair, transparent, data-driven** way to decide *who* earns *what*. Guesswork or a single "
        "raw number (like step count) isn't defensible."
    )
    st.markdown("## The FitScore solution")
    st.markdown("""
    <div class="fs-flow">
      <div class="fs-flow-step"><b>1. Wearable data</b><br>Activity + sleep from a Fitbit-style device</div>
      <div class="fs-flow-arrow">→</div>
      <div class="fs-flow-step"><b>2. Wellness score</b><br>A transparent 0–90 score (rule-based)</div>
      <div class="fs-flow-arrow">→</div>
      <div class="fs-flow-step"><b>3. ML intelligence</b><br>Persona, streak-risk, next-day forecast</div>
      <div class="fs-flow-arrow">→</div>
      <div class="fs-flow-step"><b>4. Reward tier</b><br>Transparent policy → illustrative discount</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## What the insurer gets")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers scored", f"{df['Id'].nunique()}")
    c2.metric("Behavioural personas", "4")
    c3.metric("ML models working together", "6")

    st.markdown("""
    - **Personalised, explainable rewards** — every discount comes with a plain-English reason.
    - **Early churn detection** — the model flags customers likely to disengage *before* they lapse, so the insurer can nudge them.
    - **Fairness by design** — ML provides the *intelligence*; a transparent, auditable policy makes the *decision*. The model never secretly sets a premium.
    """)

    st.info(
        "**How to read this dashboard:** the **💳 Insurance Rewards** tab is the headline product. "
        "**🧮 Predict New Customer** shows it working live on someone brand-new. The tabs after that "
        "(**Persona, Streak Risk, Forecast, Business Rules**) are the ML and policy evidence underneath."
    )
    st.caption(
        "Note: reward-discount percentages are illustrative demonstration assumptions. The public "
        "wearable dataset has no real premiums, claims, or medical costs — so FitScore provides "
        "behavioural intelligence, not actuarial pricing."
    )

with tab_overview:
    st.markdown(f"### 📊 Customer {selected_user} — activity at a glance")
    st.caption("The raw behaviour everything else is built from: how active and consistent this customer is day to day.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg daily wellness score", f'{user_df["daily_wellness_score"].mean():.1f} / 90')
    col2.metric("10K Steps Club days", int(user_df["badge_10k_steps"].sum()))
    col3.metric("Comeback Kid days", int(user_df["badge_comeback_kid"].sum()))
    fig = px.line(user_df, x="ActivityDate", y="daily_wellness_score", title="Daily wellness score")
    fig.update_traces(line_color="#e0592a", line_width=2, fill="tozeroy",
                      fillcolor="rgba(224,89,42,0.08)")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_color="#1c2333", title_font_size=16,
        margin=dict(l=10, r=10, t=44, b=10), height=340,
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(gridcolor="#eef0f4", range=[0, 92], title=None),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_leaderboard:
    st.write("Resets every ISO week so late joiners and early leaders stay on a level footing.")
    weekly = weekly_capped_points(df)
    latest = weekly[["iso_year", "iso_week"]].drop_duplicates().sort_values(
        ["iso_year", "iso_week"]
    ).iloc[-1]
    board = weekly[
        (weekly["iso_year"] == latest["iso_year"]) & (weekly["iso_week"] == latest["iso_week"])
    ].sort_values("weekly_points", ascending=False).reset_index(drop=True)
    board.index += 1
    st.dataframe(board[["Id", "weekly_points"]], use_container_width=True)

with tab_persona:
    st.write(f"**{selected_user}'s persona:** {profile.loc[selected_user, 'persona']}")
    fig = px.scatter(
        profile.reset_index(),
        x="avg_steps",
        y="avg_active_minutes",
        color="persona",
        hover_data=["Id"],
        title="Behavioural personas discovered by K-Means clustering",
        color_discrete_map={
            "Under-active": "#c0392b", "Weekend Warrior": "#b8860b",
            "Consistent Mover": "#1e8a5f", "Elite Athlete": "#4d4dae",
        },
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", font_color="#1c2333",
        title_font_size=16, margin=dict(l=10, r=10, t=44, b=10), height=420,
        xaxis=dict(gridcolor="#eef0f4", title="Avg steps"),
        yaxis=dict(gridcolor="#eef0f4", title="Avg very-active minutes"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_risk:
    st.write("**Model comparison** — 4 classifiers benchmarked on the same task, same test split:")
    comparison = pd.DataFrame(
        {
            name: {
                "Accuracy": f'{rep["accuracy"]*100:.0f}%',
                "Precision (at-risk)": f'{rep["1"]["precision"]*100:.0f}%',
                "Recall (at-risk)": f'{rep["1"]["recall"]*100:.0f}%',
                "F1 (at-risk)": f'{rep["1"]["f1-score"]:.3f}',
                "ROC-AUC": f'{rep["roc_auc"]:.3f}' if rep.get("roc_auc") is not None else "n/a",
            }
            for name, (_model, rep) in model_results.items()
        }
    ).T
    st.dataframe(comparison, use_container_width=True)
    st.caption(
        f"**{best_name}** is selected by F1-score on the at-risk class, not accuracy. "
        "The label is imbalanced (~1 in 9 user-days is at-risk), so accuracy alone rewards a model "
        "that just predicts \"not at risk\" almost every time — ROC-AUC and F1 are shown alongside "
        "it because they don't have that blind spot."
    )

    latest_row = risk_df[risk_df["Id"] == selected_user].sort_values("ActivityDate").tail(1)
    if latest_row.empty:
        st.info("Not enough recent history to score this user yet.")
    else:
        prob = risk_model.predict_proba(latest_row[feature_cols])[0][1]
        st.metric("Predicted risk of missing tomorrow's step goal", f"{prob*100:.0f}%")
        if prob > 0.5:
            st.warning("🔔 Nudge suggested — this user's streak looks at risk.")
        else:
            st.success("On track — no nudge needed.")

with tab_forecast:
    st.write(
        "**Linear Regression** predicts tomorrow's step count directly (a regression task, "
        "vs. the yes/no classifiers above) from the same lag features."
    )
    col1, col2 = st.columns(2)
    col1.metric("Mean absolute error", f"{forecast_metrics['mae']:.0f} steps")
    col2.metric("R²", f"{forecast_metrics['r2']:.2f}")

    latest_row = forecast_df[forecast_df["Id"] == selected_user].sort_values("ActivityDate").tail(1)
    if latest_row.empty:
        st.info("Not enough recent history to forecast this user yet.")
    else:
        pred_steps = forecast_model.predict(latest_row[forecast_cols])[0]
        st.metric(f"Forecast for {selected_user}'s next day", f"{pred_steps:,.0f} steps")

with tab_rewards:
    st.markdown(f"### 💳 Reward recommendation for customer {selected_user}")
    st.caption(
        "**This is the product.** For any customer, FitScore combines their wellness behaviour, "
        "consistency, and two ML signals into one reward score, then a transparent policy maps that "
        "to a tier and an illustrative premium discount — with a plain-English reason."
    )

    latest_risk_row = risk_df[risk_df["Id"] == selected_user].sort_values("ActivityDate").tail(1)
    latest_risk_probability = None
    if not latest_risk_row.empty:
        latest_risk_probability = risk_model.predict_proba(latest_risk_row[feature_cols])[0][1]

    persona = profile.loc[selected_user, "persona"]
    latest_forecast_row = forecast_df[forecast_df["Id"] == selected_user].sort_values("ActivityDate").tail(1)
    latest_forecast_steps = None
    if not latest_forecast_row.empty:
        latest_forecast_steps = forecast_model.predict(latest_forecast_row[forecast_cols])[0]

    reward = calculate_reward(
        user_df,
        persona=persona,
        risk_probability=latest_risk_probability,
        forecast_steps=latest_forecast_steps,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Reward score", f"{reward.reward_score:.1f} / 100")
    c2.metric("Recommended tier", reward.tier)
    c3.metric("Illustrative discount", f"{reward.discount_pct}%")

    st.success(f"**{reward.benefit}**")

    component_df = pd.DataFrame(
        {
            "Reward component": ["Wellness behaviour", "Consistency", "Streak safety", "Next-day forecast"],
            "Score": [reward.wellness_component, reward.consistency_component, reward.risk_component,
                      reward.forecast_component],
            "Weight": ["50%", "20%", "15%", "15%"],
        }
    )
    st.dataframe(component_df, use_container_width=True, hide_index=True)

    st.write(f"**Personalization signal:** {reward.persona}")
    if latest_forecast_steps is not None:
        st.write(f"**Linear Regression next-day forecast:** {latest_forecast_steps:,.0f} steps")
    st.info(reward.explanation)
    st.write(f"**Path to next tier:** {reward.coaching_tip}")

    with st.expander("View the illustrative reward policy"):
        policy = tier_table().rename(
            columns={
                "tier": "Tier",
                "min_score": "Minimum reward score",
                "discount_pct": "Illustrative discount %",
                "benefit": "Benefit",
            }
        )
        st.dataframe(policy, use_container_width=True, hide_index=True)

    st.warning(
        "Policy note: the 0/5/10/15% discount levels are demonstration assumptions. "
        "They must not be represented as actual insurer-approved discounts or as a prediction of claim risk."
    )

with tab_predict:
    st.subheader("🧮 Predict a new user")
    st.caption(
        "This is the point of training these models: apply the SAME fitted K-Means persona model, "
        "the SAME fitted streak-risk classifier, and the SAME fitted Linear Regression forecaster to "
        "input that was never in the training data — not a lookup against existing users."
    )

    with st.form("predict_new_user"):
        st.markdown("**Typical daily activity** (used for wellness score + persona)")
        pc1, pc2, pc3 = st.columns(3)
        in_steps = pc1.number_input("Typical daily steps", 0, 40000, 7500, step=100)
        in_very = pc1.number_input("Very active minutes", 0, 300, 20)
        in_fairly = pc2.number_input("Fairly active minutes", 0, 300, 15)
        in_lightly = pc2.number_input("Lightly active minutes", 0, 500, 180)
        in_sedentary = pc3.number_input("Sedentary minutes", 0, 1440, 950)
        in_sleep = pc3.number_input("Typical sleep (minutes)", 0, 800, 420)

        st.markdown("**Consistency** (self-reported — no history exists yet for a new user)")
        in_consistency = st.slider("% of days you'd expect to score at least 60/90", 0, 100, 60)

        st.markdown("**Recent activity** (used for streak-risk + tomorrow's forecast)")
        rc1, rc2, rc3 = st.columns(3)
        in_steps_lag1 = rc1.number_input("Yesterday's steps", 0, 40000, in_steps, step=100)
        in_steps_roll3 = rc1.number_input("3-day avg steps", 0, 40000, in_steps, step=100)
        in_steps_roll7 = rc2.number_input("7-day avg steps", 0, 40000, in_steps, step=100)
        in_sleep_lag1 = rc2.number_input("Yesterday's sleep (min)", 0, 800, in_sleep)
        in_sleep_roll3 = rc3.number_input("3-day avg sleep (min)", 0, 800, in_sleep)
        in_dow = rc3.selectbox("Day being predicted", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        wellness_score = score_single_day(in_steps, in_very, in_fairly, in_lightly, in_sleep)
        persona_pred = predict_persona(in_steps, in_very, in_sedentary, in_sleep,
                                        cluster_model, scaler, persona_remap)

        feat_row = pd.DataFrame([{
            "steps_lag1": in_steps_lag1, "steps_roll3": in_steps_roll3, "steps_roll7": in_steps_roll7,
            "sleep_lag1": in_sleep_lag1, "sleep_roll3": in_sleep_roll3, "day_of_week": dow_map[in_dow],
        }])[feature_cols]
        risk_prob = float(risk_model.predict_proba(feat_row)[0][1])
        forecast_steps = float(forecast_model.predict(feat_row[forecast_cols])[0])

        reward = calculate_reward_from_inputs(
            avg_wellness_score=wellness_score, consistency_pct=in_consistency,
            persona=persona_pred, risk_probability=risk_prob, forecast_steps=forecast_steps,
        )

        st.markdown("### Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Wellness score", f"{wellness_score:.1f} / 90")
        r2.metric("Persona", persona_pred)
        r3.metric("Streak-risk probability", f"{risk_prob*100:.0f}%")
        r4.metric("Tomorrow's forecast", f"{forecast_steps:,.0f} steps")

        c1, c2, c3 = st.columns(3)
        c1.metric("Reward score", f"{reward.reward_score:.1f} / 100")
        c2.metric("Reward tier", reward.tier)
        c3.metric("Illustrative discount", f"{reward.discount_pct}%")

        if risk_prob > 0.5:
            st.warning("🔔 This profile would be flagged for a nudge.")
        else:
            st.success("✓ This profile is on track — no nudge triggered.")

        st.info(reward.explanation)
        st.write(f"**Path to next tier:** {reward.coaching_tip}")
        st.caption(
            "Every number above came from the actual fitted models (K-Means, "
            f"{best_name}, Linear Regression) run on the inputs you just entered — "
            "nothing here is looked up from the existing 140-user dataset."
        )

with tab_rules:
    st.subheader("📐 Business rules, weights & assumptions")
    st.caption("Everything the app computes traces back to one of the rules below — nothing is hidden inside a model.")

    st.markdown("#### Wellness score (0–90, rule-based — not ML)")
    st.markdown(f"""
    <div class="fs-rule-box">
    <b>steps_points</b> = min(steps / 10,000 × 40, 40)<br>
    <b>active_points</b> = min(very_active×1.0 + fairly_active×0.6 + lightly_active×0.2, 30)<br>
    <b>sleep_points</b> = 20 if sleep is 420–540 min (7–9h), else min(sleep/480 × 15, 15)<br>
    <b>daily_wellness_score</b> = steps_points + active_points + sleep_points &nbsp; (max 90)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Reward score (0–100, business policy — combines ML outputs, is NOT itself ML)")
    st.markdown("""
    <div class="fs-rule-box">
    <b>reward_score</b> = 0.50 × wellness_component<br>
    &nbsp;&nbsp;&nbsp;&nbsp;+ 0.20 × consistency_component<br>
    &nbsp;&nbsp;&nbsp;&nbsp;+ 0.15 × streak_safety_component &nbsp;(= (1 − risk_probability) × 100, from the classifier)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;+ 0.15 × forecast_component &nbsp;(= min(forecast_steps / 10,000 × 100, 100), from Linear Regression)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Reward tiers")
    policy = tier_table().rename(columns={
        "tier": "Tier", "min_score": "Minimum reward score",
        "discount_pct": "Illustrative discount %", "benefit": "Benefit",
    })
    st.dataframe(policy, use_container_width=True, hide_index=True)
    st.caption(
        "Thresholds were calibrated against this dataset's actual reward_score distribution "
        "(quartiles), not picked as round numbers — see README.md for the exact percentiles."
    )

    st.markdown("#### Streak-risk label definition")
    st.markdown("""
    <div class="fs-rule-box">
    A day is labelled <b>at-risk</b> if next-day steps &lt; 0.7 × (trailing 3-day average steps).
    This is the ground-truth label the 4 classifiers are trained to predict — it is a rule, not
    something the model invented.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Assumptions & limitations")
    for note in [
        "The reward tiers are an illustrative demonstration policy, not a real actuarial premium "
        "calculation — this data has no insurance claims, premiums, or medical-cost information in it.",
        "K-Means persona labels (e.g. \"Weekend Warrior\") are assigned by step-count rank order; the "
        "clustering doesn't independently verify that a persona's behaviour matches its name.",
        "Consistency for the \"Predict New User\" tab is self-reported, since a brand-new user has no "
        "day-by-day history yet to compute it from — this is shown explicitly in that tab's results.",
        "Model selection uses F1-score on the at-risk class rather than raw accuracy, because the "
        "at-risk label is imbalanced (~1 in 9 user-days) — see the Streak Risk tab for the full comparison.",
    ]:
        st.markdown(f'<div class="fs-assumption">{note}</div>', unsafe_allow_html=True)

with tab_reflect:
    st.write(
        "**NLP:** a lexicon-based sentiment scorer reads your weekly reflection and tags it "
        "Positive / Neutral / Negative — the same idea behind tools like NLTK's VADER, just with "
        "a small hand-built wellness word list, so it needs no downloads."
    )
    week_label = st.text_input("Week label", value="2024-W01")
    reflection_text = st.text_area(
        "How did this week feel?",
        placeholder="e.g. Felt tired early on but got more consistent by Friday...",
    )
    if st.button("Save reflection"):
        if reflection_text.strip():
            result = score_sentiment(reflection_text)
            sql_store.save_reflection(
                str(selected_user), week_label, reflection_text, result["label"], result["score"]
            )
            st.success(f"Saved — sentiment: **{result['label']}** (score {result['score']})")
        else:
            st.warning("Write something first.")

    st.divider()
    st.write("**Past reflections (read from SQLite via SQL):**")
    history = sql_store.get_user_reflections(str(selected_user))
    if history.empty:
        st.info("No reflections saved yet for this user.")
    else:
        st.dataframe(history, use_container_width=True)

with tab_coach:
    from rag_coach import ask_fitcoach_rag, generation_available

    gen_on = generation_available()
    if gen_on:
        st.write(
            "**RAG, via LangChain + Claude:** your question retrieves the most relevant tips "
            "(TF-IDF), and those tips are passed to Claude as grounding context to *generate* a "
            "written answer to any question — full end-to-end Retrieval-Augmented Generation."
        )
        st.success("🟢 Generative mode is ON (Anthropic API key detected).")
    else:
        st.write(
            "**RAG, via LangChain:** your question is matched against the wellness-tips knowledge "
            "base using `TFIDFRetriever` (TF-IDF word-overlap ranking), so it works offline with no "
            "API key. This is the retrieval half of RAG."
        )
        st.info(
            "🔵 Retrieval-only mode (no API key). To switch on generative answers to *any* "
            "question: `pip install anthropic`, set the `ANTHROPIC_API_KEY` environment variable, "
            "and restart. See `rag_coach.py` for details."
        )

    question = st.text_input("Ask FitCoach something", placeholder="How do I improve my sleep score?")
    if question:
        result = ask_fitcoach_rag(question, retriever)
        if result["mode"] == "generative" and result.get("answer"):
            st.markdown(f"**FitCoach:** {result['answer']}")
            with st.expander("Sources (retrieved tips this answer is grounded in)"):
                for t in result["tips"]:
                    st.caption(f"[{t['topic']}] {t['text']}")
        else:
            if result.get("note"):
                st.caption(result["note"])
            for t in result["tips"]:
                st.info(f"**[{t['topic']}]** {t['text']}")

    st.divider()
    st.write("**Top users (SQL query joining scores + badges):**")
    st.dataframe(sql_store.get_top_users(limit=5), use_container_width=True)
