"""
NLP component: scores the sentiment of a user's free-text weekly reflection
("How did this week feel?"). Deliberately lexicon-based rather than a
downloaded model - it is transparent ("we count wellness-relevant positive
and negative words and take the balance"), and it needs zero downloads or
GPU, so it deploys anywhere.

Swap-in note: this is the same idea behind NLTK's VADER or TextBlob's
polarity score, just with a small hand-picked lexicon instead of a
pre-built one.
"""
import re

POSITIVE_WORDS = {
    "great", "good", "happy", "energised", "energized", "strong", "proud",
    "motivated", "productive", "rested", "refreshed", "consistent", "improved",
    "improving", "progress", "confident", "focused", "calm", "healthy", "win",
    "winning", "accomplished", "grateful", "better",
}

NEGATIVE_WORDS = {
    "tired", "exhausted", "stressed", "sick", "unwell", "lazy", "bad",
    "struggling", "struggle", "sore", "injured", "unmotivated", "anxious",
    "burnt", "burnout", "sleepless", "sluggish", "overwhelmed", "worse",
    "behind", "guilty", "frustrated", "low",
}


def score_sentiment(text: str) -> dict:
    """Returns a score in [-1, 1] and a Positive/Neutral/Negative label."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return {"score": 0.0, "label": "Neutral"}

    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = (pos - neg) / len(words)

    if score > 0.03:
        label = "Positive"
    elif score < -0.03:
        label = "Negative"
    else:
        label = "Neutral"

    return {"score": round(score, 3), "label": label}
