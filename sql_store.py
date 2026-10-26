"""
SQL storage layer. Everything the app computes gets written to a local
SQLite database (fitscore.db) and read back with SQL, instead of staying as
in-memory dataframes only - this is the project's SQL / database / ETL layer.

SQLite needs no server to install or manage, which keeps this in line with
the "implement simply" brief while still being real SQL underneath
(pandas.read_sql runs actual SQL statements against actual tables).
"""
import sqlite3
import pandas as pd
from pathlib import Path

from wellness_tips import WELLNESS_TIPS

DB_PATH = Path("fitscore.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path, check_same_thread=False)


def build_database(daily_df: pd.DataFrame, weekly_df: pd.DataFrame, db_path: Path = DB_PATH):
    """(Re)builds the database from scratch - safe to call on every app start
    since the source of truth is always the CSVs, not the DB."""
    conn = get_connection(db_path)

    daily_df.to_sql("daily_scores", conn, if_exists="replace", index=False)
    weekly_df.to_sql("weekly_leaderboard", conn, if_exists="replace", index=False)
    pd.DataFrame(WELLNESS_TIPS).to_sql("wellness_tips", conn, if_exists="replace", index=False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            week_label TEXT NOT NULL,
            reflection_text TEXT NOT NULL,
            sentiment_label TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_reflection(user_id: str, week_label: str, text: str, sentiment_label: str, sentiment_score: float,
                     db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    conn.execute(
        """INSERT INTO user_reflections (user_id, week_label, reflection_text, sentiment_label, sentiment_score)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, week_label, text, sentiment_label, sentiment_score),
    )
    conn.commit()
    conn.close()


def get_user_reflections(user_id: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = get_connection(db_path)
    df = pd.read_sql(
        "SELECT week_label, reflection_text, sentiment_label, sentiment_score, created_at "
        "FROM user_reflections WHERE user_id = ? ORDER BY created_at DESC",
        conn, params=(user_id,),
    )
    conn.close()
    return df


def get_top_users(db_path: Path = DB_PATH, limit: int = 10) -> pd.DataFrame:
    """Example analyst-style SQL query: top users by average daily score,
    joined against how many badges they've earned."""
    conn = get_connection(db_path)
    query = """
        SELECT
            Id AS user_id,
            ROUND(AVG(daily_wellness_score), 1) AS avg_score,
            SUM(badge_10k_steps) AS days_10k_club,
            SUM(badge_comeback_kid) AS comeback_days
        FROM daily_scores
        GROUP BY Id
        ORDER BY avg_score DESC
        LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return df
