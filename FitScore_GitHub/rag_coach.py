"""
"Ask FitCoach" — a Retrieval-Augmented Generation (RAG) assistant.

RAG has two halves:
  R (retrieval) — find the most relevant wellness tips for the user's question.
  G (generation) — write a natural-language answer grounded in those tips.

This module does BOTH, and degrades gracefully depending on what's available:

  MODE 1 — Full generative RAG (best).
    If an Anthropic API key is present (env var ANTHROPIC_API_KEY), the
    retrieved tips are passed to Claude as grounding context and it writes a
    real, conversational answer to ANY question — not just canned tips. This
    is genuine end-to-end RAG.

  MODE 2 — Retrieval-only (no key / offline).
    If there's no key, the assistant returns the top matching tips directly.
    Honest and free, but it can only echo existing tips.

Retrieval itself uses LangChain's TFIDFRetriever when langchain-community is
installed, and falls back to an identical scikit-learn TF-IDF + cosine
implementation otherwise, so retrieval never hard-crashes a demo.

To turn on generative mode:
    pip install anthropic
    set ANTHROPIC_API_KEY=sk-ant-...      (Windows)
    export ANTHROPIC_API_KEY=sk-ant-...   (Mac/Linux)
Get a key at https://console.anthropic.com — a small free credit is enough
for a classroom demo.
"""
import os

from wellness_tips import WELLNESS_TIPS

# --- retrieval backend: LangChain if available, else sklearn --------------
try:
    from langchain_community.retrievers import TFIDFRetriever
    from langchain_core.documents import Document
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


class _SklearnTfidfRetriever:
    """Drop-in fallback matching TFIDFRetriever's `.invoke(query)` interface,
    used only when langchain-community isn't importable. Same underlying
    algorithm (TF-IDF + cosine similarity ranking), different library."""

    class _Doc:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    def __init__(self, docs, k=3):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._docs = docs
        self._k = k
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([d.page_content for d in docs])

    def invoke(self, query: str):
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = sims.argsort()[::-1][: self._k]
        return [self._docs[i] for i in top_idx]


def build_retriever(k: int = 3):
    if _LANGCHAIN_AVAILABLE:
        docs = [Document(page_content=t["text"], metadata={"topic": t["topic"]}) for t in WELLNESS_TIPS]
        return TFIDFRetriever.from_documents(docs, k=k)
    docs = [_SklearnTfidfRetriever._Doc(t["text"], {"topic": t["topic"]}) for t in WELLNESS_TIPS]
    return _SklearnTfidfRetriever(docs, k=k)


def retrieve_tips(question: str, retriever) -> list:
    """The R in RAG: return the top-k matching tips as {text, topic} dicts.

    Pure TF-IDF struggles on very short questions ("more steps?", "unmotivated")
    because there are too few content words to match on. To make the offline
    retrieval-only mode behave sensibly, we add a light topic-keyword boost:
    if the question clearly mentions a topic, tips from that topic are pulled
    to the front. This is a small, transparent heuristic on top of TF-IDF, not
    a replacement for it — generative mode (with an API key) removes the need
    for it entirely.
    """
    results = retriever.invoke(question)
    tips = [{"text": r.page_content, "topic": r.metadata.get("topic", "general")} for r in results]

    q = question.lower()
    TOPIC_HINTS = {
        "steps": ["step", "walk", "walking", "10000", "10k"],
        "sleep": ["sleep", "sleeping", "rest", "bed", "tired", "insomnia"],
        "active": ["active minute", "cardio", "workout", "exercise", "intensity"],
        "streak": ["streak"],
        "badges": ["badge"],
        "leaderboard": ["leaderboard", "rank", "ranking"],
        "wellness_score": ["wellness score", "my score", "overall score"],
        "rewards": ["reward", "discount", "tier", "premium", "bronze", "silver", "gold", "platinum"],
        "persona": ["persona", "segment", "weekend warrior", "elite", "mover"],
        "risk": ["risk", "nudge", "at risk", "miss"],
        "motivation": ["motivat", "unmotivat", "lazy", "habit", "give up", "consistent"],
        "insurance": ["insurance", "insurer", "claim"],
    }
    hinted = [topic for topic, kws in TOPIC_HINTS.items() if any(kw in q for kw in kws)]
    if hinted:
        # pull any already-retrieved tips of a hinted topic to the front, then
        # backfill from the full corpus if the retriever missed them entirely
        from wellness_tips import WELLNESS_TIPS
        front = [t for t in tips if t["topic"] in hinted]
        if not front:
            front = [{"text": w["text"], "topic": w["topic"]}
                     for w in WELLNESS_TIPS if w["topic"] in hinted][:3]
        rest = [t for t in tips if t not in front]
        tips = (front + rest)[: len(tips) if tips else 3]
    return tips


# Backwards-compatible alias (older code / the app import this name).
def ask_fitcoach(question: str, retriever) -> list:
    return retrieve_tips(question, retriever)


def generation_available() -> bool:
    """True when both the anthropic SDK and an API key are present."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def generate_answer(question: str, tips: list, model: str = "claude-3-5-haiku-20241022") -> str:
    """
    The G in RAG: ask Claude to answer the user's question using ONLY the
    retrieved tips as grounding, so the answer stays on-topic and honest.
    Raises if generation isn't available — callers should check
    generation_available() first (the app does).
    """
    import anthropic

    context = "\n".join(f"- ({t['topic']}) {t['text']}" for t in tips)
    system = (
        "You are FitCoach, a friendly wellness assistant inside a fitness-rewards app. "
        "Answer the user's question in 2-4 short sentences using ONLY the wellness tips "
        "provided as context. If the tips don't cover the question, say so briefly and give "
        "the closest general guidance. Never invent specific medical claims, and never promise "
        "a real insurance discount — the reward tiers in this app are illustrative."
    )
    user = f"Wellness tips (context):\n{context}\n\nUser question: {question}\n\nAnswer:"

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text").strip()


def ask_fitcoach_rag(question: str, retriever) -> dict:
    """
    One-call helper the app uses. Always retrieves; additionally generates an
    answer when generation is available. Returns:
        {
          "mode": "generative" | "retrieval",
          "answer": str | None,     # natural-language answer (generative mode)
          "tips": [{text, topic}],  # the grounding tips (always present)
        }
    """
    tips = retrieve_tips(question, retriever)
    if generation_available():
        try:
            answer = generate_answer(question, tips)
            return {"mode": "generative", "answer": answer, "tips": tips}
        except Exception as e:  # network/key/quota problem -> fall back cleanly
            return {"mode": "retrieval", "answer": None, "tips": tips,
                    "note": f"(Generation unavailable, showing tips instead: {type(e).__name__})"}
    return {"mode": "retrieval", "answer": None, "tips": tips}
