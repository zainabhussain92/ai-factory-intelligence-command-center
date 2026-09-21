"""Stage IV — generation layer.

Live LLM support (Google Gemini)
---------------------------------
``generate_grounded_answer`` checks for a Google AI API key
(``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``) in the environment. If one is
set, it calls the Gemini API (via the standard library's ``urllib`` -
no extra package required) with a prompt built ONLY from the retrieved
evidence, so the model is instructed to answer strictly from what was
retrieved. Any failure (missing key, no network, API error, bad response)
is caught and the function falls back to ``_local_grounded_generation``
instead of crashing the pipeline - the app must keep working even with no
key or no network, per the Stage IV brief ("provide a local/fallback mode
so the RAG pipeline can still be tested without exposing credentials").

This is the same fallback pattern already used for Stage II's GRU and
Stage III's CV/NLP models (from-scratch local implementation, clearly
documented, because the "standard" dependency could not always be
reached).

The unsupported/no-RAG generator (`generate_unsupported_answer`) is a
*separate, explicitly labeled* function used only for the RAG-vs-no-RAG
demonstration. It deliberately has no access to the knowledge base, so it
can only answer from generic, non-factory-specific phrasing — which is
exactly the contrast Stage IV's demonstration is supposed to show.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from src.rag.retriever import RetrievedEvidence

NO_EVIDENCE_MESSAGE = (
    "The factory knowledge base does not contain sufficient information to "
    "answer this question. No retrieved document met the similarity "
    "threshold for this query."
)

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


def _try_live_llm(question: str, evidence: list[RetrievedEvidence]) -> str | None:
    """Attempt a real Gemini API call if a Google AI key is configured.
    Returns None on any failure (missing key, no network, API error) so the
    caller can fall back locally without crashing the pipeline.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    context = "\n\n".join(
        f"[{e.document} p.{e.page}{' - ' + e.section if e.section else ''}] {e.evidence}"
        for e in evidence
    )
    prompt = (
        "You are a factory maintenance assistant. Answer the question "
        "using ONLY the evidence below - do not invent procedures, "
        "thresholds, or facts that are not stated in the evidence. If the "
        "evidence is insufficient to answer, say so explicitly.\n\n"
        f"Evidence:\n{context}\n\nQuestion: {question}"
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{GEMINI_URL}?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        candidates = body.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            json.JSONDecodeError, KeyError, IndexError):
        # Missing network, invalid/expired key, rate limit, malformed
        # response, etc. - never let a live-LLM problem crash the RAG
        # pipeline; the evidence-only fallback below still answers.
        return None


def _local_grounded_generation(question: str, evidence: list[RetrievedEvidence]) -> str:
    """Deterministic, evidence-only answer synthesis (no external LLM)."""
    if not evidence:
        return NO_EVIDENCE_MESSAGE

    # Use the single most relevant chunk as the primary answer, and any
    # additional chunks as supporting detail — never inventing text beyond
    # what was retrieved.
    primary = evidence[0]
    lines = [
        f"Based on {primary.document}"
        + (f" ({primary.section})" if primary.section else "")
        + f", page {primary.page}: {primary.evidence.strip()}"
    ]
    for extra in evidence[1:]:
        lines.append(
            f"Additionally, {extra.document}"
            + (f" ({extra.section})" if extra.section else "")
            + f", page {extra.page} states: {extra.evidence.strip()}"
        )
    return " ".join(lines)


def generate_grounded_answer(question: str, evidence: list[RetrievedEvidence]) -> dict:
    """Return {"answer": str, "used_live_llm": bool}."""
    live_answer = _try_live_llm(question, evidence)
    if live_answer is not None:
        return {"answer": live_answer, "used_live_llm": True}
    return {"answer": _local_grounded_generation(question, evidence), "used_live_llm": False}


def generate_unsupported_answer(question: str) -> str:
    """Generic answer with NO access to the factory knowledge base.

    Used only for the RAG-vs-no-RAG demonstration (see
    src/rag/demo_rag_vs_no_rag.py). This intentionally cannot cite any
    factory-specific threshold, document, or procedure — that is the point
    of the comparison.
    """
    return (
        "In general, if a machine shows a concerning sensor reading, "
        "standard practice is to reduce load, monitor the situation, and "
        "consult the manufacturer's documentation or a qualified "
        "technician before deciding whether to continue operating the "
        "equipment. (This answer is generated without access to this "
        "factory's specific manuals or SOPs, so it cannot state exact "
        "thresholds, required response steps, or which document to follow.)"
    )
