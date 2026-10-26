"""
Insurer-style reward eligibility engine for FitScore.

Important: this is a transparent business-policy layer, not an actuarial model.
The public Fitbit dataset contains activity/sleep behaviour but no premiums,
claims, medical costs, or insurer loss data. Therefore, the discount values are
illustrative policy assumptions, while the user's reward score is computed from
observed behaviour plus the ML streak-risk probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


# Illustrative insurer-style reward policy for a classroom/demo application.
# The score thresholds are business rules, not ML-learned values, but they
# were calibrated by actually running the pipeline on the project dataset
# and inspecting the resulting reward_score distribution rather than picked
# blindly. On this data reward_score has quartiles at roughly 66 / 73 / 79
# (25th/50th/75th percentile); the four tiers below track those quartiles,
# so each tier holds a meaningful, non-empty share of the population instead
# of everyone clustering into one or two bands. If you regenerate the
# dataset or change the wellness-score formula, re-run run_pipeline_test.py
# and re-check these cutoffs against the new distribution printed there.
REWARD_TIERS = [
    {"tier": "Platinum", "min_score": 79, "discount_pct": 15, "benefit": "15% illustrative premium discount + wellness perks"},
    {"tier": "Gold", "min_score": 73, "discount_pct": 10, "benefit": "10% illustrative premium discount + wellness perks"},
    {"tier": "Silver", "min_score": 66, "discount_pct": 5, "benefit": "5% illustrative premium discount + wellness perks"},
    {"tier": "Bronze", "min_score": 0, "discount_pct": 0, "benefit": "No illustrative premium discount; baseline wellness benefits"},
]

PERSONA_BENEFITS = {
    "Under-active": "Guided activity challenge and starter wellness resources",
    "Weekend Warrior": "Consistency challenge rewards and weekly activity perks",
    "Consistent Mover": "Sleep/wellness vouchers and consistency perks",
    "Elite Athlete": "Advanced wellness perks and premium reward eligibility",
}


@dataclass
class RewardResult:
    reward_score: float
    tier: str
    discount_pct: int
    benefit: str
    wellness_component: float
    consistency_component: float
    risk_component: float
    forecast_component: float
    persona: str
    explanation: str
    coaching_tip: str


def _tier_for_score(score: float) -> Dict[str, object]:
    for tier in REWARD_TIERS:
        if score >= tier["min_score"]:
            return tier
    return REWARD_TIERS[-1]


def _coaching_tip(reward_score: float, wellness: float, consistency: float,
                   risk: float, forecast: float) -> str:
    """
    "How to reach the next tier" — this is what turns the reward score from a
    static grade into something the user can act on, per the design brief:
    tell them the next tier and which component has the most room to move it.

    Room-to-move is headroom weighted by each component's share of the total
    score (0.50/0.20/0.15/0.15), so this naturally points at whichever gap is
    actually worth the most points to close, not just whichever number looks
    lowest.
    """
    tiers_sorted = sorted(REWARD_TIERS, key=lambda t: t["min_score"])
    next_tier = next((t for t in tiers_sorted if t["min_score"] > reward_score), None)

    headroom = {
        "wellness behaviour": 0.50 * (100 - wellness),
        "consistency": 0.20 * (100 - consistency),
        "streak safety": 0.15 * (100 - risk),
        "next-day forecast": 0.15 * (100 - forecast),
    }
    focus = max(headroom, key=headroom.get)

    if next_tier is None:
        return f"Already at the top tier (Platinum) — keep {focus} steady to stay there."
    points_needed = round(next_tier["min_score"] - reward_score, 1)
    return f"{points_needed} points from {next_tier['tier']}. Biggest opportunity: {focus}."


def _finalize_reward(wellness_component: float, consistency_component: float,
                      risk_component: float, forecast_component: float,
                      persona: str, note: str = "") -> "RewardResult":
    """
    Shared scoring core used by both calculate_reward() (existing users, from
    their full daily history) and calculate_reward_from_inputs() (a
    hypothetical/new user, from summary inputs) — both paths land here so the
    weighting, tier lookup, explanation, and coaching tip logic exist in
    exactly one place rather than being duplicated and risking drift between
    the two entry points.
    """
    reward_score = round(
        0.50 * wellness_component
        + 0.20 * consistency_component
        + 0.15 * risk_component
        + 0.15 * forecast_component,
        1,
    )

    tier = _tier_for_score(reward_score)
    persona_benefit = PERSONA_BENEFITS.get(persona, "Wellness programme benefits")
    explanation = (
        f"Average wellness score contributes {wellness_component:.0f}/100, "
        f"consistency contributes {consistency_component:.0f}/100, "
        f"streak safety contributes {risk_component:.0f}/100, and "
        f"the next-day forecast contributes {forecast_component:.0f}/100. "
        f"The combined reward score is {reward_score:.1f}/100.{note}"
    )
    benefit = f"{tier['benefit']}; personalised benefit: {persona_benefit}"
    coaching_tip = _coaching_tip(reward_score, wellness_component, consistency_component,
                                  risk_component, forecast_component)

    return RewardResult(
        reward_score=reward_score,
        tier=str(tier["tier"]),
        discount_pct=int(tier["discount_pct"]),
        benefit=benefit,
        wellness_component=round(wellness_component, 1),
        consistency_component=round(consistency_component, 1),
        risk_component=round(risk_component, 1),
        forecast_component=round(forecast_component, 1),
        persona=persona,
        explanation=explanation,
        coaching_tip=coaching_tip,
    )


def calculate_reward(
    user_df: pd.DataFrame,
    persona: str,
    risk_probability: Optional[float] = None,
    forecast_steps: Optional[float] = None,
) -> RewardResult:
    """
    Convert an EXISTING user's observed behaviour into a transparent reward
    score.

    Components:
      - 50% average wellness score (normalised from the existing 0-90 score)
      - 20% consistency (% observed days scoring at least 60/90)
      - 15% streak safety (1 - ML predicted streak-risk probability)
      - 15% next-day forecast strength (Linear Regression forecast vs 10,000 steps)

    The result is a reward-policy recommendation, not a real insurance risk score.
    """
    if user_df.empty:
        raise ValueError("No activity data available for this user.")

    avg_wellness = float(user_df["daily_wellness_score"].mean())
    wellness_component = max(0.0, min(100.0, (avg_wellness / 90.0) * 100.0))

    consistency_component = float(
        (user_df["daily_wellness_score"] >= 60).mean() * 100.0
    )

    if risk_probability is None or pd.isna(risk_probability):
        risk_probability = 0.0
    risk_probability = max(0.0, min(1.0, float(risk_probability)))
    risk_component = (1.0 - risk_probability) * 100.0

    if forecast_steps is None or pd.isna(forecast_steps):
        forecast_component = wellness_component
    else:
        forecast_component = max(0.0, min(100.0, (float(forecast_steps) / 10_000.0) * 100.0))

    return _finalize_reward(wellness_component, consistency_component,
                             risk_component, forecast_component, persona)


def calculate_reward_from_inputs(
    avg_wellness_score: float,
    consistency_pct: float,
    persona: str,
    risk_probability: Optional[float] = None,
    forecast_steps: Optional[float] = None,
) -> RewardResult:
    """
    Same weighting as calculate_reward(), but for a NEW/hypothetical user who
    has no day-by-day history to average over yet — takes the two behavioural
    inputs (average wellness score, consistency %) directly instead of
    deriving them from a dataframe. This is what powers the "Predict New
    User" tab: a real, live application of the fitted persona/risk/forecast
    models to input nobody has seen before, not a lookup against existing
    users.

    avg_wellness_score: expected on the same 0-90 scale add_wellness_points()
    produces (pass the score computed for the user's typical/entered day).
    consistency_pct: 0-100, self-reported since there's no history to compute
    it from - this is surfaced explicitly in the UI as self-reported input,
    not something the model inferred.
    """
    wellness_component = max(0.0, min(100.0, (float(avg_wellness_score) / 90.0) * 100.0))
    consistency_component = max(0.0, min(100.0, float(consistency_pct)))

    if risk_probability is None or pd.isna(risk_probability):
        risk_probability = 0.0
    risk_probability = max(0.0, min(1.0, float(risk_probability)))
    risk_component = (1.0 - risk_probability) * 100.0

    if forecast_steps is None or pd.isna(forecast_steps):
        forecast_component = wellness_component
    else:
        forecast_component = max(0.0, min(100.0, (float(forecast_steps) / 10_000.0) * 100.0))

    note = (" (new-user estimate: consistency is self-reported, not computed "
            "from day-by-day history like it is for existing users.)")
    return _finalize_reward(wellness_component, consistency_component,
                             risk_component, forecast_component, persona, note=note)


def tier_table() -> pd.DataFrame:
    """Return the illustrative reward policy as a DataFrame for the dashboard."""
    return pd.DataFrame(REWARD_TIERS)[["tier", "min_score", "discount_pct", "benefit"]]
