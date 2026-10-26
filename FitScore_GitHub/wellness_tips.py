"""
Knowledge base for the 'Ask FitCoach' RAG feature.

Kept as plain Python so it's trivial to read, extend, or swap for a CSV later.
Each tip is tagged with a topic so you can filter/report on which topics get
asked about most.

This is the retrieval corpus. When Ask FitCoach runs in generative mode
(an Anthropic API key is available), the most relevant tips here are passed
to the model as grounding context and it writes a natural-language answer to
ANY question. When it runs in retrieval-only mode (no key / offline), the
most relevant tips are returned directly. Either way, a richer corpus means
better answers, so this list is intentionally broad across every topic the
app touches: steps, sleep, active minutes, streaks, badges, the leaderboard,
the wellness score, the reward tiers, personas, and the insurance framing.
"""

WELLNESS_TIPS = [
    # --- steps ------------------------------------------------------------
    {"topic": "steps", "text": "To increase your steps and walk more, add a short 10-15 minute walk after each meal - this is the easiest way to raise your daily step count without a dedicated workout."},
    {"topic": "steps", "text": "If you want more steps, take walking phone calls, use the stairs instead of the lift, and park further away. Small incidental movement adds up fast."},
    {"topic": "steps", "text": "Breaking a big step goal into three short walks across the day makes it far easier to hit than one long session."},
    {"topic": "steps", "text": "A daily step target of around 8,000-10,000 steps is a strong wellness goal. Increase it gradually rather than all at once so the habit sticks."},
    {"topic": "steps", "text": "If your step count has been falling, lower your target for one week rather than skipping entirely - a reduced but achievable goal protects the habit and your streak."},
    {"topic": "steps", "text": "Morning walks are the most reliable because they happen before the day gets busy. Evening step goals are the first thing dropped when work overruns."},
    # --- sleep ------------------------------------------------------------
    {"topic": "sleep", "text": "To improve your sleep score, aim for 7-9 hours a night. Consistent sleep is the single biggest lever on your overall wellness score."},
    {"topic": "sleep", "text": "For better sleep quality, go to bed and wake up at the same time every day, even on weekends - regular timing helps more than total hours alone."},
    {"topic": "sleep", "text": "Avoid caffeine within eight hours of bed and screens for the last thirty minutes. Poor sleep is the strongest predictor of a low-activity day that follows."},
    {"topic": "sleep", "text": "If you wake often, keep the bedroom cool and dark. Time in bed well above time asleep usually points to sleep quality rather than quantity."},
    {"topic": "sleep", "text": "A short wind-down routine - dim lights, no phone, a few minutes of reading - signals your body to sleep and improves how rested you feel."},
    # --- active minutes ---------------------------------------------------
    {"topic": "active", "text": "To earn more active minutes, add short bursts of brisk activity. Even a fast 10-minute walk counts toward your very-active and fairly-active minutes."},
    {"topic": "active", "text": "Very-active minutes are weighted most heavily in your wellness score, so a little vigorous activity is worth a lot of gentle activity."},
    {"topic": "active", "text": "Cardio like a brisk walk, cycling, or a jog raises active minutes faster than light pottering around - intensity matters for this metric."},
    # --- sedentary --------------------------------------------------------
    {"topic": "sedentary", "text": "Sitting for more than 900 minutes a day is linked to worse health outcomes even if you exercise. Stand and move for two minutes every hour."},
    {"topic": "sedentary", "text": "Breaking up long sitting stretches with brief movement is one of the easiest wins for both your health and your activity minutes."},
    # --- streaks ----------------------------------------------------------
    {"topic": "streak", "text": "To protect your activity streak, log even light activity on busy or unwell days. The streak rewards consistency over intensity."},
    {"topic": "streak", "text": "If you break a streak, restart the same day rather than waiting for a fresh week. A short walk is enough to get going again."},
    {"topic": "streak", "text": "Streaks work because they make progress visible. Pairing your walk with something you already do daily, like a podcast, makes the habit stick."},
    # --- badges -----------------------------------------------------------
    {"topic": "badges", "text": "Badges reward milestones. One unlocks on any high-activity day and resets daily; the Comeback badge celebrates bouncing back after a quiet stretch."},
    {"topic": "badges", "text": "Badges are designed to reward both peak days and recovery, so a comeback after a slow week is celebrated just like a personal best."},
    # --- leaderboard ------------------------------------------------------
    {"topic": "leaderboard", "text": "The leaderboard resets every ISO week (Monday to Sunday), so a slow start doesn't cost you the rest of the month."},
    {"topic": "leaderboard", "text": "Weekly leaderboard points are capped, so one enormous day can't be farmed into an unbeatable score. Steady daily effort wins over the week."},
    # --- wellness score ---------------------------------------------------
    {"topic": "wellness_score", "text": "Your wellness score is out of 90 and comes from three parts: steps (up to 40), active minutes (up to 30), and sleep (up to 20)."},
    {"topic": "wellness_score", "text": "Improving whichever of steps, active minutes, or sleep is currently lowest is the fastest way to raise your overall wellness score."},
    {"topic": "wellness_score", "text": "Each part of the wellness score is capped, so you can't max it on steps alone - a balanced day of movement and sleep scores highest."},
    # --- rewards / tiers --------------------------------------------------
    {"topic": "rewards", "text": "Your reward score is out of 100 and maps to a tier: Bronze, Silver, Gold, or Platinum. Higher tiers carry a larger illustrative premium discount."},
    {"topic": "rewards", "text": "The reward score combines your wellness behaviour (50%), consistency (20%), streak safety (15%), and next-day activity forecast (15%)."},
    {"topic": "rewards", "text": "To reach a higher reward tier and a bigger discount, focus on consistent daily activity and good sleep rather than occasional intense bursts."},
    {"topic": "rewards", "text": "Because consistency is weighted heavily, a steady month of moderate activity earns more reward than a few spectacular days and many empty ones."},
    {"topic": "rewards", "text": "The discount percentages (0/5/10/15%) are illustrative programme assumptions for this demo, not a real insurer-approved premium discount."},
    # --- personas ---------------------------------------------------------
    {"topic": "persona", "text": "Personas are behavioural segments found by clustering - Under-active, Weekend Warrior, Consistent Mover, and Elite Athlete - and they personalise which benefits you're offered."},
    {"topic": "persona", "text": "Your persona doesn't set your discount directly. It personalises your rewards, while your reward score sets the tier - two separate things."},
    {"topic": "persona", "text": "A Weekend Warrior gains most from adding one weekday session; a Consistent Mover benefits from a harder target. Benefits are tailored to the persona."},
    # --- streak-risk / nudges --------------------------------------------
    {"topic": "risk", "text": "The app predicts your risk of missing tomorrow's activity goal and nudges you before a streak breaks. Acting on an early nudge is the point."},
    {"topic": "risk", "text": "A day is flagged 'at risk' when tomorrow's activity is predicted to fall below 70% of your recent three-day average - a heads-up, not a penalty."},
    # --- insurance framing ------------------------------------------------
    {"topic": "insurance", "text": "Insurers run wellness-reward programmes because sustained activity and good sleep are linked to lower long-term health risk."},
    {"topic": "insurance", "text": "Reward banding is recalculated each quarter from your average engagement, so a strong final month can lift your tier."},
    # --- motivation / general --------------------------------------------
    {"topic": "motivation", "text": "Set a weekly goal rather than a daily one. Weekly goals absorb a single bad day without the sense of failure that ends a habit."},
    {"topic": "motivation", "text": "Habit stacking - attaching a new habit to one you already have - beats willpower for long-term consistency."},
    {"topic": "nutrition", "text": "Staying hydrated makes walking and light exercise feel easier, which indirectly supports your step and active-minute goals."},
    {"topic": "stress", "text": "Short breathing exercises lower perceived stress. Stressful weeks show the sharpest drops in activity, so managing stress protects your streak."},
]
