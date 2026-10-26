# FitScore — Gamified Wellness & Insurance Rewards

**Business story:** wearable activity + sleep data → cleaned & merged →
rule-based wellness score → ML-driven personalization (persona, streak-risk,
next-day forecast) → transparent insurer-style reward tier → personalized
Streamlit dashboard.

```
Fitbit activity + sleep CSVs
        │  data_pipeline.py  (merge, dedupe, fillna)
        ▼
   Master daily table  ── wellness score + badges (rule-based)
        │
        ├──► K-Means (ml_models.py) ─────────► persona (4 segments)
        ├──► 4-classifier benchmark ─────────► streak-risk probability
        ├──► Linear Regression ──────────────► next-day step forecast
        │
        ▼
   reward_engine.py  (business-policy layer, NOT ML)
   wellness(50%) + consistency(20%) + streak-safety(15%) + forecast(15%)
        │
        ▼
   Reward score → tier (Platinum/Gold/Silver/Bronze) → illustrative discount
        │
        ▼
   app.py — gamified Streamlit app (6 tabs), SQLite persistence, NLP + RAG
```

## What's ML here (six algorithms)

| Task | Type | Algorithm(s) |
|---|---|---|
| Persona discovery | Unsupervised clustering | **K-Means** (4 personas) |
| Streak-risk prediction | Classification (4-way benchmark) | **Logistic Regression, Decision Tree, Random Forest, Naive Bayes** |
| Next-day steps forecast | Regression | **Linear Regression** |

Plus: **NLP** (lexicon sentiment on weekly reflections), **LangChain TF-IDF
retrieval** (Ask FitCoach), **SQL/SQLite** persistence.

**Tech stack:** Python, Pandas, NumPy, scikit-learn, Plotly, Streamlit,
LangChain, SQLite.

## Data source

The pipeline (`data_pipeline.py`) expects exactly the schema of Kaggle's
**"FitBit Fitness Tracker Data"** (`arashnic/fitbit`, CC0 public domain):
`dailyActivity_merged.csv` and `sleepDay_merged.csv` in `data/`.

Two ways to fill `data/`, both schema-compatible with zero code changes:

1. **Real data.** Download from
   https://www.kaggle.com/datasets/arashnic/fitbit and drop the two CSVs
   into `data/`. This is the real, well-known, 30-user dataset.
2. **Synthetic data (what ships in this folder).** `generate_synthetic_fitbit.py`
   simulates 140 users × 120 days with the *identical* column names, calibrated
   to published summary statistics for the real file (mean steps ~7,600,
   mean sedentary ~991 min, mean sleep ~419 min, etc.), with realistic gaps
   (sleep logged for ~67% of user-days, not every day — the real file has
   this too). Included so the repo runs out of the box. Regenerate anytime with:
   `python generate_synthetic_fitbit.py`

Swapping between them is a file-copy, not a code change — every column name
matches.

## Actual results (from `run_pipeline_test.py`, this synthetic run)

Dataset: **140 users × 120 days** (16,800 daily rows), synthetic but
calibrated to the real Fitbit schema and its published summary statistics.

**K-Means personas** — 4 segments, all non-trivially populated:

| Persona | Users | Avg steps | Avg sleep (logged nights only) |
|---|---|---|---|
| Elite Athlete | 17 | 14,277 | 435 min |
| Consistent Mover | 35 | 9,215 | 393 min |
| Weekend Warrior | 34 | 7,328 | 462 min |
| Under-active | 54 | 5,805 | 405 min |

**Streak-risk classifier benchmark** (label: next day's steps < 70% of the
trailing 3-day average; 8.2% of user-days are labelled at-risk):

| Model | Accuracy | Precision (at-risk) | Recall (at-risk) | F1 (at-risk) | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 90.8% | 21.4% | 4.4% | 0.073 | 0.532 |
| Random Forest | 70.5% | 12.8% | 44.6% | 0.199 | 0.605 |
| **Logistic Regression** | 62.9% | 12.4% | **58.0%** | **0.205** | **0.635** |
| Decision Tree | 52.6% | 9.9% | 58.6% | 0.169 | 0.567 |

**Selected model: Logistic Regression — by F1 on the at-risk class, not
accuracy.** Why:

- The label is imbalanced — only ~8% of user-days are "at-risk". A model that
  simply predicts "not at risk" every single time scores ~91% accuracy
  (that's exactly what Naive Bayes is doing here: 90.8% accuracy but 4.4%
  recall — it catches almost none of the users it exists to catch).
- The entire purpose of this model is to fire a nudge *before* someone breaks
  their streak. Missing an at-risk user (a false negative) is the expensive
  error; sending an occasional unnecessary nudge (a false positive) is cheap.
- Logistic Regression catches 58% of true at-risk cases. Its accuracy looks
  low precisely because it's willing to raise false alarms to avoid missing
  people — the correct trade-off for a churn-prevention nudge.
- **ROC-AUC** (a threshold-independent measure of how well a model *ranks*
  at-risk users above safe ones) confirms this isn't cherry-picking:
  Logistic Regression has the best AUC (0.635) of the four as well, so it's
  genuinely the best-calibrated ranker, not just the F1 winner.

A single accuracy number is the *wrong* headline for an imbalanced problem.

**Linear Regression (next-day step forecast):** MAE ≈ 1,485 steps, R² ≈
0.70 (vs. a naive "predict yesterday's 3-day average" baseline of ≈1,575
steps MAE — the model beats the naive baseline). The R² improved from ~0.53
in an earlier 65-user build after adding a 7-day rolling-steps and a 3-day
rolling-sleep feature and enlarging the dataset.

**Reward tiers** (recalibrated against this dataset's actual score
distribution — see `reward_engine.py` header comment):

| Tier | Threshold | Users (of 140) | Illustrative discount |
|---|---|---|---|
| Platinum | ≥79 | 47 | 15% |
| Gold | 73–78.9 | 26 | 10% |
| Silver | 66–72.9 | 30 | 5% |
| Bronze | <66 | 37 | 0% |

## The two apps in this repo

**`app.py` — member-facing gamified app (the main one).** A member logs in
with their ID and sees outcomes, not model names: a fitness score, XP level,
streak, badges and reward tier. Six tabs:

- **🏠 My Fitness** — today's snapshot, level, streak, streak-risk warning in plain English
- **📈 Progress** — wellness-score and step trends
- **🏆 Challenges & Rewards** — badges, active challenges, reward tier and discount
- **🎮 What-If** — enter any made-up day and the same fitted models (K-Means,
  Logistic Regression, Linear Regression) score it live: fitness score,
  persona, streak outlook, next-day steps and reward tier
- **📝 Reflection** — short check-ins scored with NLP sentiment, shown as mood
- **🤖 Ask Fit Coach** — TF-IDF retrieval over wellness tips (optional
  generative answers if `ANTHROPIC_API_KEY` is set)

Gamification logic (XP, levels, streaks, badges, challenges) lives in
`gamification.py`.

**`app_admin_backup.py` — analyst / admin dashboard (earlier build).** Shows
the technical view: classifier benchmark, personas, forecast, leaderboard,
plus these two tabs:

- **🧮 Predict New Customer** — enter any hypothetical person's activity/sleep
  numbers and the app runs the *same fitted* K-Means, Logistic Regression,
  and Linear Regression models against that input to return a persona,
  streak-risk probability, next-day forecast, wellness score, and full
  reward recommendation. This is the "the model learned a pattern and applies
  it to unseen input" story made concrete — it is real inference on new
  input, not a lookup against existing users. (In the offline HTML preview,
  the fitted model coefficients are exported and the identical math runs in
  the browser; this was verified to match the Python models to
  floating-point precision.)
- **📐 Business Rules** — a dedicated tab spelling out every formula, weight,
  threshold, and assumption in one place: the 0–90 wellness formula, the
  50/20/15/15 reward weighting, the tier cut-offs, the streak-risk label
  definition, and the honest limitations. Nothing the app outputs is hidden
  inside a model.

## What actually moved the numbers (and what didn't)

Accuracy can't be manufactured — you can only add real signal or measure
more fairly. What was tried, and the *measured* effect:

- **More data + better features (kept):** 65→140 users, 91→120 days, plus a
  7-day rolling-steps and 3-day rolling-sleep feature. Measurably helped the
  forecast (R² 0.53 → 0.70). Classification F1 stayed in a narrow band.
- **Reducing day-to-day noise (tested, then reverted):** lowering simulation
  noise *looked* like it should make the pattern easier to learn, but it made
  classification F1 *worse* — because the at-risk label is partly defined by
  day-to-day variation, so less noise means fewer at-risk days to learn from.
  Reverted, with the finding documented in `generate_synthetic_fitbit.py`.
- **Fairer metric (added):** ROC-AUC now sits alongside accuracy/F1 so the
  model comparison can't be misread through the lens of accuracy alone.

Takeaway: 62.9% accuracy on a deliberately imbalanced,
noisy behavioural-prediction task with a 58% at-risk recall is a *defensible,
honest* result. A 90%+ accuracy that catches 4% of at-risk users would look
better on paper and be worse in every way that matters.

## Bugs found and fixed while running this end-to-end

1. **Persona `avg_sleep` was corrupted by the scoring rule.**
   `build_master_table()` zero-fills nights with no sleep log — a correct
   *scoring* rule (no log, no sleep points). But `build_user_profile()` was
   averaging that same zero-filled column for clustering, so a user who
   simply doesn't wear their device at night looked identical to one who
   genuinely sleeps 3 hours. Fixed by tracking `sleep_logged` before the fill
   and having the persona profile average only logged nights.
2. **"Best" classifier was picked by accuracy on an imbalanced label**, which
   silently selected the model with the worst recall on the class that
   matters. Switched selection to F1 on the at-risk class, and added ROC-AUC
   to the benchmark for a threshold-independent cross-check.
3. **Reward tier thresholds left Bronze essentially empty** under the
   original 80/65/50 cutoffs. Recalibrated to the dataset's actual quartiles
   so all four tiers are meaningfully populated.

None of the six ML algorithms, the wellness-scoring formula, the
0.7×-trailing-average streak-risk label definition, the SQL layer, the NLP
approach, or the RAG approach were changed in concept — the feature set was
extended (extra rolling features) and the above three issues were fixed.

## Honest limitations

- Fitbit activity/sleep data has no insurance claims, premiums, or medical
  costs in it. The reward tiers are a demonstration policy layer, not an
  actuarial model.
- The "Weekend Warrior" / "Elite Athlete" persona *names* are just labels
  assigned by step-count rank order; the clustering itself doesn't verify
  that a "Weekend Warrior" actually spikes on weekends specifically.
- The streak-risk classifier is a genuinely hard, imbalanced task — 62.9%
  accuracy with 58% at-risk recall is the honest ceiling here, not a bug to
  be tuned away.
- The synthetic dataset is calibrated to published summary statistics, not
  drawn from the real 30 users — swap in the real Kaggle CSVs to run on the
  literal published dataset.

## Run it

Run every command from inside the project folder (the code reads `data/`
and writes `fitscore.db` relative to it).

```bash
pip install -r requirements.txt
python generate_synthetic_fitbit.py     # only if data/ is empty
streamlit run app.py                    # member app → http://localhost:8501
streamlit run app_admin_backup.py       # optional: analyst/admin dashboard
```

`fitscore.db` (SQLite) is created automatically on first run, so it is not
committed.

To re-verify the ML layer without Streamlit (what was used to validate this
build):

```bash
python run_pipeline_test.py
```

To regenerate the offline dashboard preview after any change:

```bash
python export_dashboard_data.py
```
