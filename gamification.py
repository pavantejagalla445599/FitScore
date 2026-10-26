"""
Gamification / presentation layer.

This module is the TRANSLATOR between the backend intelligence (ML models,
business rules) and what the user actually sees. The user should never see
"Logistic Regression predicted 0.78" — they should see "Streak Risk: Medium,
here's what to do about it."

Nothing here re-computes ML or business logic. It only converts already-computed
outputs into levels, XP, streaks, badges, and plain-English messages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Levels & XP
# ---------------------------------------------------------------------------
# XP is earned from the daily wellness score the user already earns. Levels are
# just XP thresholds, so "levelling up" is a direct reward for sustained
# healthy behaviour rather than a separate invented metric.
XP_PER_LEVEL = 1500

LEVEL_TITLES = [
    "Rookie", "Mover", "Strider", "Pacer", "Achiever",
    "Champion", "Elite", "Legend",
]


def compute_xp(user_df: pd.DataFrame) -> int:
    """Total XP = sum of every daily wellness score the user has earned."""
    return int(user_df["daily_wellness_score"].sum())


def level_info(xp: int) -> dict:
    level = int(xp // XP_PER_LEVEL) + 1
    into_level = xp % XP_PER_LEVEL
    title = LEVEL_TITLES[min(level - 1, len(LEVEL_TITLES) - 1)]
    return {
        "level": level,
        "title": title,
        "xp_into_level": int(into_level),
        "xp_needed": XP_PER_LEVEL,
        "pct": round(100 * into_level / XP_PER_LEVEL, 1),
        "xp_to_next": int(XP_PER_LEVEL - into_level),
    }


# ---------------------------------------------------------------------------
# Streaks
# ---------------------------------------------------------------------------
STREAK_THRESHOLD = 60  # daily wellness score that "counts" as an active day


def streak_info(user_df: pd.DataFrame, threshold: int = STREAK_THRESHOLD) -> dict:
    """Current and best run of consecutive qualifying days."""
    s = user_df.sort_values("ActivityDate")["daily_wellness_score"] >= threshold
    vals = s.tolist()

    current = 0
    for v in reversed(vals):
        if v:
            current += 1
        else:
            break

    best = run = 0
    for v in vals:
        run = run + 1 if v else 0
        best = max(best, run)

    return {"current": current, "best": best, "threshold": threshold,
            "active_days": int(s.sum()), "total_days": len(vals)}


# ---------------------------------------------------------------------------
# Risk → plain English
# ---------------------------------------------------------------------------
def risk_band(prob: float | None) -> dict:
    """Turn a model probability into a user-facing band + advice.

    The probability itself comes from the classifier; this function only
    decides how to TALK about it. Thresholds are presentation choices.
    """
    if prob is None:
        return {"band": "Unknown", "emoji": "❓", "color": "#8892a6",
                "message": "Not enough recent activity to read your streak yet.",
                "action": "Log a few more days to unlock streak insights."}
    if prob < 0.35:
        return {"band": "Low", "emoji": "🟢", "color": "#22c55e",
                "message": "Your streak looks safe. Recent activity has been steady.",
                "action": "Keep doing what you're doing — consistency is paying off."}
    if prob < 0.60:
        return {"band": "Medium", "emoji": "🟡", "color": "#f59e0b",
                "message": "Your activity has softened compared with your recent average.",
                "action": "Hit today's step goal to protect your streak."}
    return {"band": "High", "emoji": "🔴", "color": "#ef4444",
            "message": "Your recent activity has dropped sharply — your streak is at risk.",
            "action": "A 15-minute walk today is enough to get back on track."}


# ---------------------------------------------------------------------------
# Badges / achievements
# ---------------------------------------------------------------------------
def earned_badges(user_df: pd.DataFrame, streak: dict, lvl: dict) -> list[dict]:
    """Badges the user has actually earned, computed from their real data."""
    tenk = int(user_df["badge_10k_steps"].sum())
    comeback = int(user_df["badge_comeback_kid"].sum())
    best_day = float(user_df["daily_wellness_score"].max())
    avg_sleep = float(user_df.loc[user_df["sleep_logged"], "TotalMinutesAsleep"].mean()) \
        if user_df["sleep_logged"].any() else 0.0
    total_steps = int(user_df["TotalSteps"].sum())

    defs = [
        {"icon": "👟", "name": "10K Club", "desc": f"{tenk} days above 10,000 steps",
         "earned": tenk >= 1, "progress": min(tenk / 10, 1.0), "goal": "10 days"},
        {"icon": "🔥", "name": "Streak Keeper", "desc": f"Best streak: {streak['best']} days",
         "earned": streak["best"] >= 7, "progress": min(streak["best"] / 7, 1.0), "goal": "7-day streak"},
        {"icon": "💪", "name": "Comeback Kid", "desc": f"Bounced back {comeback} times",
         "earned": comeback >= 1, "progress": min(comeback / 3, 1.0), "goal": "3 comebacks"},
        {"icon": "😴", "name": "Rested", "desc": f"Average sleep {avg_sleep:.0f} min",
         "earned": avg_sleep >= 420, "progress": min(avg_sleep / 420, 1.0), "goal": "7h average"},
        {"icon": "⭐", "name": "Perfect Day", "desc": f"Best day scored {best_day:.0f}/90",
         "earned": best_day >= 85, "progress": min(best_day / 85, 1.0), "goal": "85+ in a day"},
        {"icon": "🚀", "name": "Millionaire", "desc": f"{total_steps:,} steps walked",
         "earned": total_steps >= 1_000_000, "progress": min(total_steps / 1_000_000, 1.0),
         "goal": "1M steps"},
        {"icon": "🏆", "name": f"Level {lvl['level']}", "desc": f"Reached {lvl['title']}",
         "earned": lvl["level"] >= 2, "progress": min(lvl["level"] / 5, 1.0), "goal": "Level 5"},
    ]
    return defs


# ---------------------------------------------------------------------------
# Challenges (derived from real behaviour, not invented)
# ---------------------------------------------------------------------------
def active_challenges(user_df: pd.DataFrame, streak: dict) -> list[dict]:
    recent = user_df.sort_values("ActivityDate").tail(7)
    steps_7d = int(recent["TotalSteps"].sum())
    active_min_7d = int((recent["VeryActiveMinutes"] + recent["FairlyActiveMinutes"]).sum())
    good_sleep_nights = int((recent["TotalMinutesAsleep"] >= 420).sum())

    return [
        {"icon": "🏃", "name": "Weekly 50K", "desc": "Walk 50,000 steps this week",
         "current": steps_7d, "target": 50_000, "unit": "steps"},
        {"icon": "⚡", "name": "Active Hour", "desc": "150 active minutes this week",
         "current": active_min_7d, "target": 150, "unit": "min"},
        {"icon": "🌙", "name": "Sleep Well", "desc": "5 nights of 7h+ sleep",
         "current": good_sleep_nights, "target": 5, "unit": "nights"},
        {"icon": "🔥", "name": "Keep the Flame", "desc": "Reach a 7-day streak",
         "current": streak["current"], "target": 7, "unit": "days"},
    ]


# ---------------------------------------------------------------------------
# Sentiment trend → friendly summary
# ---------------------------------------------------------------------------
def mood_summary(entries: list[dict]) -> dict:
    """entries: [{'label': 'Positive'|'Neutral'|'Negative', 'score': float}, ...]"""
    if not entries:
        return {"headline": "No reflections yet", "emoji": "📝",
                "detail": "Write your first weekly check-in to start tracking how you feel.",
                "counts": {"Positive": 0, "Neutral": 0, "Negative": 0}}

    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for e in entries:
        counts[e["label"]] = counts.get(e["label"], 0) + 1

    dominant = max(counts, key=counts.get)
    avg = float(np.mean([e["score"] for e in entries]))

    if dominant == "Positive":
        headline, emoji = "Your reflections are mostly positive", "😊"
        detail = "You're sounding motivated and consistent lately — that mindset usually shows up in your activity too."
    elif dominant == "Negative":
        headline, emoji = "You've been having a tough stretch", "💙"
        detail = "Low weeks are normal. Small, easy wins tend to rebuild momentum faster than big goals."
    else:
        headline, emoji = "Your reflections are steady and balanced", "😌"
        detail = "No big highs or lows — a good base to build a stronger habit from."

    return {"headline": headline, "emoji": emoji, "detail": detail,
            "counts": counts, "avg": round(avg, 3)}


# ---------------------------------------------------------------------------
# Today snapshot
# ---------------------------------------------------------------------------
def today_snapshot(user_df: pd.DataFrame) -> dict:
    latest = user_df.sort_values("ActivityDate").iloc[-1]
    return {
        "date": latest["ActivityDate"],
        "steps": int(latest["TotalSteps"]),
        "calories": int(latest["Calories"]),
        "active_minutes": int(latest["VeryActiveMinutes"] + latest["FairlyActiveMinutes"]),
        "sleep_minutes": int(latest["TotalMinutesAsleep"]),
        "score": float(latest["daily_wellness_score"]),
        "step_goal": 10_000,
        "step_pct": min(100.0, round(100 * latest["TotalSteps"] / 10_000, 1)),
    }
