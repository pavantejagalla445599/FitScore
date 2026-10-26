"""
FitScore — user-facing gamified wellness app.

DESIGN PRINCIPLE
----------------
This is the PRESENTATION layer. All the intelligence (K-Means personas, the
4-classifier streak-risk benchmark, Linear Regression forecasting, lexicon
sentiment analysis, TF-IDF retrieval) still runs underneath exactly as before
— see ml_models.py, reward_engine.py, nlp_sentiment.py, rag_coach.py.

What changed is that the user never sees model names, accuracy, precision or
probabilities. They see outcomes: a score, a level, a streak, a risk warning
in plain English, and what to do next. The technical evidence lives in the
code and in run_pipeline_test.py, not cluttering the interface.
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import (
    build_master_table, add_wellness_points, weekly_capped_points,
    assign_badges, score_single_day,
)
from ml_models import (
    build_user_profile, train_persona_clusters, predict_persona,
    build_streak_risk_dataset, train_streak_risk_models,
    build_steps_forecast_dataset, train_steps_forecast_model,
)
from nlp_sentiment import score_sentiment
from rag_coach import build_retriever
import sql_store
from reward_engine import calculate_reward, tier_table
import gamification as g

st.set_page_config(page_title="FitScore", page_icon="🏃", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Dark-blue gamified theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root{
  --bg:#0b1220; --bg2:#111c33; --card:#16233d; --card2:#1b2b49;
  --line:#24365c; --ink:#eaf0ff; --muted:#9fb0d0;
  --accent:#00e5a0; --accent2:#37b6ff; --warn:#ffb020; --bad:#ff5d6c; --purple:#a06bff;
}
.stApp{ background:linear-gradient(180deg,#0b1220 0%,#0d1526 100%); color:var(--ink); }
html,body,[class*="css"]{ color:var(--ink); }
h1,h2,h3,h4,h5,p,span,div,label,li{ color:var(--ink); }
section.main > div{ padding-top:6px; }
#MainMenu,footer{ visibility:hidden; }

.fs-hero{
  background:linear-gradient(135deg,#132244 0%,#1b3060 55%,#233b73 100%);
  border:1px solid var(--line); border-radius:18px; padding:22px 26px; margin-bottom:16px;
  box-shadow:0 10px 30px rgba(0,0,0,.35);
}
.fs-hero h1{ margin:0; font-size:26px; letter-spacing:-.3px; }
.fs-hero p{ margin:6px 0 0; color:var(--muted); font-size:13.5px; }

.card{
  background:linear-gradient(160deg,var(--card) 0%,var(--card2) 100%);
  border:1px solid var(--line); border-radius:16px; padding:18px 20px; margin-bottom:14px;
  box-shadow:0 6px 18px rgba(0,0,0,.25);
}
.card h4{ margin:0 0 10px; font-size:14px; color:var(--muted); font-weight:600;
  text-transform:uppercase; letter-spacing:.06em; }
.big{ font-size:34px; font-weight:800; line-height:1.1; }
.sub{ color:var(--muted); font-size:12.5px; margin-top:4px; }

.stat{ background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:14px 16px; text-align:center; height:100%; }
.stat .ico{ font-size:20px; }
.stat .val{ font-size:22px; font-weight:800; margin-top:4px; }
.stat .lab{ color:var(--muted); font-size:11.5px; margin-top:2px; text-transform:uppercase;
  letter-spacing:.05em; }

.bar{ background:#0d1830; border-radius:99px; height:10px; overflow:hidden; margin:8px 0 4px; }
.bar > div{ height:100%; border-radius:99px;
  background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.bar.warn > div{ background:linear-gradient(90deg,#ffb020,#ff8a3d); }
.bar.bad  > div{ background:linear-gradient(90deg,#ff5d6c,#ff3d80); }

.badge{ display:inline-flex; align-items:center; gap:8px; background:var(--card2);
  border:1px solid var(--line); border-radius:12px; padding:10px 12px; width:100%;
  margin-bottom:8px; }
.badge.on{ border-color:var(--accent); box-shadow:0 0 0 1px rgba(0,229,160,.25) inset; }
.badge .bi{ font-size:20px; }
.badge .bn{ font-weight:700; font-size:13.5px; }
.badge .bd{ color:var(--muted); font-size:11.5px; }
.badge.off{ opacity:.5; }

.pill{ display:inline-block; padding:4px 12px; border-radius:99px; font-size:12px;
  font-weight:700; }
.pill.low{ background:rgba(34,197,94,.15); color:#4ade80; }
.pill.med{ background:rgba(245,158,11,.15); color:#fbbf24; }
.pill.high{ background:rgba(239,68,68,.15); color:#f87171; }

.tierbox{ border-radius:16px; padding:18px 20px; border:1px solid var(--line);
  background:linear-gradient(135deg,#16233d,#1d3055); }

/* streamlit widget theming */
button[data-baseweb="tab"]{ font-size:14px; font-weight:700; color:var(--muted); }
button[data-baseweb="tab"][aria-selected="true"]{ color:var(--accent) !important; }
div[data-baseweb="tab-highlight"]{ background:var(--accent) !important; height:3px; }
div[data-baseweb="tab-border"]{ background:var(--line); }
.stButton>button,.stFormSubmitButton>button{
  background:linear-gradient(90deg,var(--accent),var(--accent2)); color:#06131f;
  border:none; border-radius:10px; font-weight:800; padding:9px 22px; }
.stButton>button:hover,.stFormSubmitButton>button:hover{ filter:brightness(1.08); color:#06131f; }
div[data-baseweb="select"]>div{ background:var(--card); border-color:var(--line); color:var(--ink); }
div[data-baseweb="input"] input,textarea{ background:var(--card) !important; color:var(--ink) !important;
  border-color:var(--line) !important; }
div[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; }
div[data-testid="stAlert"]{ border-radius:12px; }
.stTextInput label,.stTextArea label,.stSelectbox label,.stNumberInput label,.stSlider label{
  color:var(--muted) !important; font-size:12.5px !important; }
hr{ border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend (cached) — unchanged intelligence, just loaded once
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = build_master_table()
    df = add_wellness_points(df)
    return assign_badges(df)


@st.cache_data(show_spinner=False)
def run_ml(df):
    profile = build_user_profile(df)
    profile, cluster_model, scaler, remap = train_persona_clusters(profile)
    risk_df, fcols = build_streak_risk_dataset(df)
    results, best = train_streak_risk_models(risk_df, fcols)
    fdf, fccols = build_steps_forecast_dataset(df)
    fmodel, fmetrics = train_steps_forecast_model(fdf, fccols)
    return (profile, cluster_model, scaler, remap, risk_df, fcols,
            results, best, results[best][0], results[best][1],
            fdf, fccols, fmodel, fmetrics)


@st.cache_resource(show_spinner=False)
def get_retriever():
    return build_retriever(k=3)


df = load_data()
(profile, cluster_model, scaler, remap, risk_df, feature_cols,
 model_results, best_name, risk_model, report,
 forecast_df, forecast_cols, forecast_model, forecast_metrics) = run_ml(df)

weekly_all = weekly_capped_points(df)
sql_store.build_database(df, weekly_all)
retriever = get_retriever()

PERSONA_BLURB = {
    "Elite Athlete": ("🏅", "You train hard and consistently. Your challenge is recovery, not effort."),
    "Consistent Mover": ("🎯", "You show up reliably. Small intensity boosts will push you to the next level."),
    "Weekend Warrior": ("⚡", "You go big on weekends. Adding one weekday session would transform your score."),
    "Under-active": ("🌱", "You're early in the journey. Short daily walks build momentum fastest."),
}

# ---------------------------------------------------------------------------
# LOGIN
# ---------------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown("""
    <div class="fs-hero" style="text-align:center;padding:44px 26px;">
      <h1 style="font-size:40px;">🏃 FitScore</h1>
      <p style="font-size:15px;">Your personal fitness game. Move more, earn rewards,
      unlock better insurance benefits.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.3, 1])
    with c2:
        st.markdown('<div class="card"><h4>Sign in</h4>', unsafe_allow_html=True)
        uid = st.selectbox("Choose your member ID", sorted(df["Id"].unique()))
        if st.button("🔓  Log in", use_container_width=True):
            st.session_state.user = uid
            st.rerun()
        st.markdown('<div class="sub">Demo mode — pick any member ID to explore that profile.</div></div>',
                    unsafe_allow_html=True)

    a, b, c = st.columns(3)
    for col, ico, t, d in [
        (a, "🔥", "Build streaks", "Stay active daily and watch your streak grow."),
        (b, "🏆", "Earn badges", "Unlock achievements as you hit milestones."),
        (c, "💳", "Get rewarded", "Healthy habits unlock better premium benefits."),
    ]:
        col.markdown(f'<div class="stat"><div class="ico">{ico}</div>'
                     f'<div class="val" style="font-size:15px;">{t}</div>'
                     f'<div class="lab" style="text-transform:none;letter-spacing:0;">{d}</div></div>',
                     unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# LOGGED IN — compute this member's picture
# ---------------------------------------------------------------------------
uid = st.session_state.user
user_df = df[df["Id"] == uid].sort_values("ActivityDate")

xp = g.compute_xp(user_df)
lvl = g.level_info(xp)
streak = g.streak_info(user_df)
today = g.today_snapshot(user_df)
persona = profile.loc[uid, "persona"]

r_row = risk_df[risk_df["Id"] == uid].sort_values("ActivityDate").tail(1)
risk_prob = float(risk_model.predict_proba(r_row[feature_cols])[0][1]) if not r_row.empty else None
risk = g.risk_band(risk_prob)

f_row = forecast_df[forecast_df["Id"] == uid].sort_values("ActivityDate").tail(1)
forecast_steps = float(forecast_model.predict(f_row[forecast_cols])[0]) if not f_row.empty else None

reward = calculate_reward(user_df, persona=persona,
                          risk_probability=risk_prob, forecast_steps=forecast_steps)
badges = g.earned_badges(user_df, streak, lvl)
challenges = g.active_challenges(user_df, streak)

# header
h1, h2 = st.columns([4, 1])
with h1:
    p_ico, p_text = PERSONA_BLURB.get(persona, ("🎯", ""))
    st.markdown(f"""
    <div class="fs-hero">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
        <div>
          <h1>👋 Welcome back, Member {uid}</h1>
          <p>{p_ico} {persona} · {p_text}</p>
        </div>
        <div style="text-align:right;">
          <div style="font-size:12px;color:#9fb0d0;text-transform:uppercase;letter-spacing:.08em;">Level {lvl['level']}</div>
          <div style="font-size:26px;font-weight:800;color:#00e5a0;">{lvl['title']}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
with h2:
    st.write("")
    if st.button("Log out", use_container_width=True):
        st.session_state.user = None
        st.rerun()

tab_home, tab_prog, tab_rew, tab_whatif, tab_ref, tab_coach = st.tabs(
    ["🏠  My Fitness", "📈  Progress", "🏆  Challenges & Rewards",
     "🎮  What-If", "📝  Reflection", "🤖  Ask Fit Coach"]
)

# ===========================================================================
# 1 — MY FITNESS
# ===========================================================================
with tab_home:
    left, right = st.columns([1.15, 1])

    with left:
        ring = go.Figure(go.Pie(
            values=[today["score"], 90 - today["score"]], hole=.76,
            marker_colors=["#00e5a0", "#16233d"], textinfo="none",
            hoverinfo="skip", sort=False, showlegend=False))
        ring.update_layout(
            height=250, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(text=f"<b>{today['score']:.0f}</b>", x=.5, y=.55,
                     font=dict(size=44, color="#eaf0ff"), showarrow=False),
                dict(text="FITNESS SCORE / 90", x=.5, y=.36,
                     font=dict(size=11, color="#9fb0d0"), showarrow=False)])
        st.markdown('<div class="card"><h4>Today</h4>', unsafe_allow_html=True)
        st.plotly_chart(ring, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="sub" style="text-align:center;">Fitness Score = Activity + Sleep + Consistency</div>'
            f'</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="card">
          <h4>Level progress</h4>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <div class="big" style="color:#37b6ff;">Lv {lvl['level']}</div>
            <div class="sub">{lvl['xp_into_level']} / {lvl['xp_needed']} XP</div>
          </div>
          <div class="bar"><div style="width:{lvl['pct']}%"></div></div>
          <div class="sub">{lvl['xp_to_next']} XP to reach <b>Level {lvl['level']+1}</b></div>
        </div>
        <div class="card">
          <h4>Streak</h4>
          <div style="display:flex;align-items:center;gap:14px;">
            <div style="font-size:40px;">🔥</div>
            <div>
              <div class="big">{streak['current']} days</div>
              <div class="sub">Personal best: {streak['best']} days</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    for col, ico, val, lab in [
        (s1, "👟", f"{today['steps']:,}", "Steps today"),
        (s2, "🔥", f"{today['calories']:,}", "Calories"),
        (s3, "⚡", f"{today['active_minutes']}", "Active minutes"),
        (s4, "😴", f"{today['sleep_minutes']//60}h {today['sleep_minutes']%60}m", "Sleep"),
    ]:
        col.markdown(f'<div class="stat"><div class="ico">{ico}</div>'
                     f'<div class="val">{val}</div><div class="lab">{lab}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    g1, g2 = st.columns([1, 1])
    with g1:
        st.markdown(f"""
        <div class="card">
          <h4>Daily step goal</h4>
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <div class="big">{today['step_pct']:.0f}%</div>
            <div class="sub">{today['steps']:,} / {today['step_goal']:,}</div>
          </div>
          <div class="bar"><div style="width:{min(today['step_pct'],100)}%"></div></div>
          <div class="sub">{'🎉 Goal smashed!' if today['step_pct'] >= 100 else f"{today['step_goal']-today['steps']:,} steps to go today."}</div>
        </div>""", unsafe_allow_html=True)
    with g2:
        pill = {"Low": "low", "Medium": "med", "High": "high"}.get(risk["band"], "med")
        st.markdown(f"""
        <div class="card">
          <h4>Streak watch</h4>
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="font-size:22px;">{risk['emoji']}</span>
            <span class="pill {pill}">Risk: {risk['band']}</span>
          </div>
          <div style="font-size:13.5px;">{risk['message']}</div>
          <div class="sub" style="margin-top:8px;">👉 {risk['action']}</div>
        </div>""", unsafe_allow_html=True)

    fc = f"{forecast_steps:,.0f}" if forecast_steps else "—"
    st.markdown(f"""
    <div class="card">
      <h4>What's next</h4>
      <div style="display:flex;gap:26px;flex-wrap:wrap;">
        <div><div class="sub">Tomorrow you're on track for</div>
             <div style="font-size:22px;font-weight:800;color:#37b6ff;">{fc} steps</div></div>
        <div><div class="sub">Your reward tier</div>
             <div style="font-size:22px;font-weight:800;color:#00e5a0;">{reward.tier}</div></div>
        <div><div class="sub">Path to next tier</div>
             <div style="font-size:13.5px;margin-top:6px;">{reward.coaching_tip}</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ===========================================================================
# 2 — PROGRESS
# ===========================================================================
with tab_prog:
    d = user_df.copy()
    d["week"] = d["ActivityDate"].dt.to_period("W").dt.start_time

    recent = d.tail(30)
    prev = d.tail(60).head(30)
    delta = recent["daily_wellness_score"].mean() - prev["daily_wellness_score"].mean() if len(prev) else 0
    arrow = "▲" if delta >= 0 else "▼"
    dcol = "#00e5a0" if delta >= 0 else "#ff5d6c"

    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="stat"><div class="ico">📊</div><div class="val">{recent["daily_wellness_score"].mean():.0f}</div>'
                f'<div class="lab">Avg score (30d)</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="stat"><div class="ico">{arrow}</div>'
                f'<div class="val" style="color:{dcol};">{abs(delta):.1f}</div>'
                f'<div class="lab">vs previous 30d</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="stat"><div class="ico">✅</div>'
                f'<div class="val">{streak["active_days"]}/{streak["total_days"]}</div>'
                f'<div class="lab">Active days</div></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h4>Fitness score over time</h4>', unsafe_allow_html=True)
    f1 = px.line(d, x="ActivityDate", y="daily_wellness_score")
    f1.update_traces(line_color="#00e5a0", line_width=2.5, fill="tozeroy",
                     fillcolor="rgba(0,229,160,.10)",
                     hovertemplate="%{x|%d %b}<br><b>%{y:.0f}</b> / 90<extra></extra>")
    f1.update_layout(height=280, margin=dict(l=6, r=6, t=6, b=6),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font_color="#9fb0d0",
                     xaxis=dict(showgrid=False, title=None),
                     yaxis=dict(gridcolor="#1d2c4a", title="Score / 90", range=[0, 95]))
    st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="sub">Higher is better. A rising line means your activity, '
                'sleep and consistency are improving together.</div></div>', unsafe_allow_html=True)

    cA, cB = st.columns(2)
    with cA:
        wk = d.groupby("week", as_index=False)["TotalSteps"].mean().tail(12)
        st.markdown('<div class="card"><h4>Average daily steps, by week</h4>', unsafe_allow_html=True)
        f2 = px.bar(wk, x="week", y="TotalSteps")
        f2.update_traces(marker_color="#37b6ff",
                         hovertemplate="Week of %{x|%d %b}<br><b>%{y:,.0f}</b> steps/day<extra></extra>")
        f2.update_layout(height=250, margin=dict(l=6, r=6, t=6, b=6),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#9fb0d0", xaxis=dict(showgrid=False, title=None),
                         yaxis=dict(gridcolor="#1d2c4a", title="Steps/day"))
        st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="sub">Taller bars = more active weeks.</div></div>', unsafe_allow_html=True)
    with cB:
        sl = d[d["sleep_logged"]].tail(30)
        st.markdown('<div class="card"><h4>Sleep, last 30 nights</h4>', unsafe_allow_html=True)
        f3 = px.line(sl, x="ActivityDate", y="TotalMinutesAsleep")
        f3.update_traces(line_color="#a06bff", line_width=2.5,
                         hovertemplate="%{x|%d %b}<br><b>%{y:.0f}</b> min<extra></extra>")
        f3.add_hline(y=420, line_dash="dot", line_color="#9fb0d0",
                     annotation_text="7h target", annotation_font_color="#9fb0d0")
        f3.update_layout(height=250, margin=dict(l=6, r=6, t=6, b=6),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font_color="#9fb0d0", xaxis=dict(showgrid=False, title=None),
                         yaxis=dict(gridcolor="#1d2c4a", title="Minutes asleep"))
        st.plotly_chart(f3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="sub">Nights above the dotted line hit the 7-hour target.</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="card"><h4>Consistency map — last 8 weeks</h4>', unsafe_allow_html=True)
    hd = d.tail(56).copy()
    hd["dow"] = hd["ActivityDate"].dt.day_name().str[:3]
    hd["wk"] = hd["ActivityDate"].dt.isocalendar().week.astype(int)
    pivot = hd.pivot_table(index="dow", columns="wk", values="daily_wellness_score", aggfunc="mean")
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex([o for o in order if o in pivot.index])
    f4 = px.imshow(pivot, color_continuous_scale=["#16233d", "#1d5e57", "#00e5a0"],
                   aspect="auto", labels=dict(color="Score"))
    f4.update_traces(hovertemplate="%{y}, week %{x}<br><b>%{z:.0f}</b>/90<extra></extra>")
    f4.update_layout(height=230, margin=dict(l=6, r=6, t=6, b=6),
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font_color="#9fb0d0", xaxis_title="Week number", yaxis_title=None,
                     coloraxis_colorbar=dict(title=None))
    st.plotly_chart(f4, use_container_width=True, config={"displayModeBar": False})
    st.markdown('<div class="sub">Brighter squares are stronger days — look for dark gaps '
                'to spot the days of the week you tend to skip.</div></div>', unsafe_allow_html=True)

# ===========================================================================
# 3 — CHALLENGES & REWARDS
# ===========================================================================
with tab_rew:
    tier_col = {"Platinum": "#a06bff", "Gold": "#ffb020",
                "Silver": "#9fb0d0", "Bronze": "#c98a5b"}.get(reward.tier, "#00e5a0")
    st.markdown(f"""
    <div class="tierbox">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:18px;">
        <div>
          <div class="sub" style="text-transform:uppercase;letter-spacing:.08em;">Your reward tier</div>
          <div style="font-size:38px;font-weight:800;color:{tier_col};">{reward.tier}</div>
          <div class="sub">Reward score {reward.reward_score:.0f} / 100</div>
        </div>
        <div style="text-align:right;">
          <div class="sub" style="text-transform:uppercase;letter-spacing:.08em;">Premium benefit</div>
          <div style="font-size:38px;font-weight:800;color:#00e5a0;">{reward.discount_pct}%</div>
          <div class="sub">illustrative discount</div>
        </div>
      </div>
      <div class="bar" style="margin-top:14px;"><div style="width:{reward.reward_score}%"></div></div>
      <div class="sub">🎯 {reward.coaching_tip}</div>
    </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h4>How your reward is calculated</h4>', unsafe_allow_html=True)
    for label, val, why in [
        ("💚 Healthy behaviour", reward.wellness_component, "Your activity and sleep scores"),
        ("📅 Consistency", reward.consistency_component, "How often you hit a good day"),
        ("🔥 Streak safety", reward.risk_component, "How likely you are to stay on track"),
        ("🔮 Momentum", reward.forecast_component, "Where your activity is heading next"),
    ]:
        cls = "" if val >= 65 else ("warn" if val >= 40 else "bad")
        st.markdown(f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;font-size:13.5px;">
            <span>{label}</span><span style="font-weight:700;">{val:.0f}/100</span>
          </div>
          <div class="bar {cls}"><div style="width:{val}%"></div></div>
          <div class="sub">{why}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("❓ Why do I get this tier?"):
        st.markdown(f"""
Your **Reward Score** blends four things, weighted by how much they matter:

| What it measures | Weight |
|---|---|
| 💚 Healthy behaviour (activity + sleep) | 50% |
| 📅 Consistency (sustained effort) | 20% |
| 🔥 Streak safety (staying on track) | 15% |
| 🔮 Momentum (where you're heading) | 15% |

That score maps to a tier: **Bronze → Silver → Gold → Platinum**. Consistency
matters more than any single great day — steady effort beats occasional bursts.

*Discount percentages shown are illustrative programme assumptions for this demo.*
        """)

    st.write("")
    st.markdown('<div class="card"><h4>Active challenges</h4>', unsafe_allow_html=True)
    cc = st.columns(2)
    for i, ch in enumerate(challenges):
        pct = min(100, 100 * ch["current"] / ch["target"])
        done = pct >= 100
        cls = "" if pct >= 60 else ("warn" if pct >= 30 else "bad")
        cc[i % 2].markdown(f"""
        <div style="background:#1b2b49;border:1px solid {'#00e5a0' if done else '#24365c'};
                    border-radius:12px;padding:12px 14px;margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;">
            <span style="font-weight:700;font-size:13.5px;">{ch['icon']} {ch['name']}</span>
            <span style="font-size:12px;color:#9fb0d0;">{ch['current']:,}/{ch['target']:,} {ch['unit']}</span>
          </div>
          <div class="bar {cls}"><div style="width:{pct}%"></div></div>
          <div class="sub">{'✅ Complete!' if done else ch['desc']}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h4>Achievements</h4>', unsafe_allow_html=True)
    bc = st.columns(3)
    for i, b in enumerate(badges):
        on = "on" if b["earned"] else "off"
        extra = "" if b["earned"] else f'<div class="bar" style="margin-top:6px;"><div style="width:{b["progress"]*100:.0f}%"></div></div><div class="sub">Goal: {b["goal"]}</div>'
        bc[i % 3].markdown(f"""
        <div class="badge {on}">
          <span class="bi">{b['icon']}</span>
          <span style="flex:1;">
            <div class="bn">{b['name']} {'✓' if b['earned'] else ''}</div>
            <div class="bd">{b['desc']}</div>{extra}
          </span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# 4 — WHAT-IF  (live prediction on a made-up day — the ML, gamified)
# ===========================================================================
with tab_whatif:
    st.markdown('<div class="card"><h4>🎮 Try a "what-if" day</h4>'
                '<div class="sub">Enter any typical day and see what fitness score, level, '
                'streak outlook and reward tier it would earn — instantly. Great for setting a '
                'goal: tweak the numbers and watch your tier change.</div></div>',
                unsafe_allow_html=True)

    with st.form("whatif"):
        st.markdown("**🏃 A typical day**")
        a, b, c = st.columns(3)
        wi_steps = a.slider("Steps", 0, 25000, 8000, 250)
        wi_very = a.slider("Intense active minutes", 0, 120, 20)
        wi_fairly = b.slider("Moderate active minutes", 0, 120, 15)
        wi_lightly = b.slider("Light active minutes", 0, 400, 180, 10)
        wi_sed = c.slider("Sitting minutes", 0, 1440, 950, 10)
        wi_sleep = c.slider("Sleep (minutes)", 0, 720, 420, 10)

        st.markdown("**📅 Recent form** (how the last few days have gone)")
        d, e, f = st.columns(3)
        wi_lag1 = d.slider("Yesterday's steps", 0, 25000, 8000, 250)
        wi_roll3 = d.slider("Last 3 days avg steps", 0, 25000, 8000, 250)
        wi_roll7 = e.slider("Last 7 days avg steps", 0, 25000, 8000, 250)
        wi_sleep3 = e.slider("Last 3 days avg sleep (min)", 0, 720, 420, 10)
        wi_consistency = f.slider("How often you hit a good day (%)", 0, 100, 65, 5)
        wi_dow = f.selectbox("Day of week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        go_pred = st.form_submit_button("✨  See my results", use_container_width=True)

    if go_pred:
        from reward_engine import calculate_reward_from_inputs

        dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        # --- real fitted models run on the entered day ---
        wi_score = score_single_day(wi_steps, wi_very, wi_fairly, wi_lightly, wi_sleep)
        wi_persona = predict_persona(wi_steps, wi_very, wi_sed, wi_sleep,
                                     cluster_model, scaler, remap)
        feat = pd.DataFrame([{
            "steps_lag1": wi_lag1, "steps_roll3": wi_roll3, "steps_roll7": wi_roll7,
            "sleep_lag1": wi_sleep3, "sleep_roll3": wi_sleep3, "day_of_week": dow_map[wi_dow],
        }])[feature_cols]
        wi_riskp = float(risk_model.predict_proba(feat)[0][1])
        wi_forecast = float(forecast_model.predict(feat[forecast_cols])[0])
        wi_risk = g.risk_band(wi_riskp)
        wi_reward = calculate_reward_from_inputs(
            avg_wellness_score=wi_score, consistency_pct=wi_consistency,
            persona=wi_persona, risk_probability=wi_riskp, forecast_steps=wi_forecast)
        wi_lvl = g.level_info(int(wi_score * 30))  # rough season-worth of XP at this level

        st.markdown("### ✨ Your what-if results")
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f'<div class="stat"><div class="ico">💯</div>'
                    f'<div class="val">{wi_score:.0f}/90</div><div class="lab">Fitness score</div></div>',
                    unsafe_allow_html=True)
        pico = {"Elite Athlete": "🏅", "Consistent Mover": "🎯",
                "Weekend Warrior": "⚡", "Under-active": "🌱"}.get(wi_persona, "🎯")
        r2.markdown(f'<div class="stat"><div class="ico">{pico}</div>'
                    f'<div class="val" style="font-size:15px;">{wi_persona}</div>'
                    f'<div class="lab">Your type</div></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="stat"><div class="ico">{wi_risk["emoji"]}</div>'
                    f'<div class="val" style="font-size:17px;">{wi_risk["band"]}</div>'
                    f'<div class="lab">Streak outlook</div></div>', unsafe_allow_html=True)
        r4.markdown(f'<div class="stat"><div class="ico">🔮</div>'
                    f'<div class="val">{wi_forecast:,.0f}</div><div class="lab">Next-day steps</div></div>',
                    unsafe_allow_html=True)

        tier_col = {"Platinum": "#a06bff", "Gold": "#ffb020",
                    "Silver": "#9fb0d0", "Bronze": "#c98a5b"}.get(wi_reward.tier, "#00e5a0")
        st.markdown(f"""
        <div class="tierbox" style="margin-top:6px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
            <div>
              <div class="sub" style="text-transform:uppercase;letter-spacing:.08em;">This day would earn</div>
              <div style="font-size:34px;font-weight:800;color:{tier_col};">{wi_reward.tier} tier</div>
            </div>
            <div style="text-align:right;">
              <div class="sub" style="text-transform:uppercase;letter-spacing:.08em;">Premium benefit</div>
              <div style="font-size:34px;font-weight:800;color:#00e5a0;">{wi_reward.discount_pct}%</div>
              <div class="sub">illustrative discount</div>
            </div>
          </div>
          <div class="bar" style="margin-top:12px;"><div style="width:{wi_reward.reward_score}%"></div></div>
          <div class="sub">Reward score {wi_reward.reward_score:.0f}/100 · 🎯 {wi_reward.coaching_tip}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="card" style="margin-top:12px;">{wi_risk["emoji"]} '
                    f'<b>Streak outlook:</b> {wi_risk["message"]}<br>'
                    f'<span class="sub">👉 {wi_risk["action"]}</span></div>', unsafe_allow_html=True)

        st.caption("Every number here is produced live by the same models that power your real "
                   "profile — run on the day you just invented, not looked up from another member.")

# ===========================================================================
# 5 — REFLECTION  (NLP sentiment, presented as mood)
# ===========================================================================
with tab_ref:
    key = f"reflections_{uid}"
    if key not in st.session_state:
        st.session_state[key] = []

    st.markdown('<div class="card"><h4>How did this week feel?</h4>', unsafe_allow_html=True)
    txt = st.text_area("Write a short check-in", height=110, label_visibility="collapsed",
                       placeholder="e.g. Felt tired early in the week but got more consistent by Friday...")
    if st.button("Save check-in"):
        if txt.strip():
            res = score_sentiment(txt)
            st.session_state[key].insert(0, {"text": txt.strip(), **res})
            sql_store.save_reflection(str(uid), str(pd.Timestamp.today().date()),
                                      txt.strip(), res["label"], res["score"])
            st.success("Saved — thanks for checking in!")
        else:
            st.warning("Write something first.")
    st.markdown('</div>', unsafe_allow_html=True)

    entries = st.session_state[key]
    mood = g.mood_summary(entries)

    st.markdown(f"""
    <div class="card">
      <h4>Your mood trend</h4>
      <div style="display:flex;align-items:center;gap:16px;">
        <div style="font-size:44px;">{mood['emoji']}</div>
        <div>
          <div style="font-size:19px;font-weight:800;">{mood['headline']}</div>
          <div class="sub" style="font-size:13px;">{mood['detail']}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if entries:
        cnt = mood["counts"]
        mc = st.columns(3)
        for col, lab, ico, colr in [(mc[0], "Positive", "😊", "#00e5a0"),
                                     (mc[1], "Neutral", "😌", "#37b6ff"),
                                     (mc[2], "Negative", "💙", "#ff5d6c")]:
            col.markdown(f'<div class="stat"><div class="ico">{ico}</div>'
                         f'<div class="val" style="color:{colr};">{cnt.get(lab,0)}</div>'
                         f'<div class="lab">{lab} check-ins</div></div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card"><h4>Past check-ins</h4>', unsafe_allow_html=True)
        for e in entries[:8]:
            ico = {"Positive": "😊", "Neutral": "😌", "Negative": "💙"}[e["label"]]
            st.markdown(f"""
            <div style="background:#1b2b49;border:1px solid #24365c;border-radius:10px;
                        padding:10px 14px;margin-bottom:8px;">
              <span style="font-size:16px;">{ico}</span>
              <span style="font-size:13.5px;margin-left:8px;">{e['text'][:160]}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Write your first check-in above to start tracking how you feel week to week.")

# ===========================================================================
# 6 — ASK FIT COACH
# ===========================================================================
with tab_coach:
    from rag_coach import ask_fitcoach_rag, generation_available

    st.markdown('<div class="card"><h4>🤖 Ask Fit Coach</h4>'
                '<div class="sub">Your personal wellness assistant — ask anything about '
                'steps, sleep, streaks, rewards or staying motivated.</div>', unsafe_allow_html=True)

    quick = ["How do I get more steps?", "How can I improve my sleep?",
             "How do I reach the next reward tier?", "I feel unmotivated"]
    qcols = st.columns(len(quick))
    picked = None
    for i, q in enumerate(quick):
        if qcols[i].button(q, use_container_width=True, key=f"q{i}"):
            picked = q

    typed = st.text_input("Ask your own question", placeholder="Type a question…",
                          label_visibility="collapsed")
    question = picked or typed
    st.markdown('</div>', unsafe_allow_html=True)

    if question:
        res = ask_fitcoach_rag(question, retriever)
        st.markdown(f'<div class="card"><h4>You asked</h4>'
                    f'<div style="font-size:14px;">“{question}”</div></div>', unsafe_allow_html=True)
        if res["mode"] == "generative" and res.get("answer"):
            st.markdown(f"""
            <div class="card" style="border-color:#00e5a0;">
              <h4>Fit Coach says</h4>
              <div style="font-size:14.5px;line-height:1.65;">{res['answer']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><h4>Fit Coach says</h4>', unsafe_allow_html=True)
            for t in res["tips"]:
                st.markdown(f"""
                <div style="background:#1b2b49;border-left:3px solid #00e5a0;border-radius:10px;
                            padding:12px 14px;margin-bottom:9px;font-size:13.5px;line-height:1.6;">
                  {t['text']}
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
