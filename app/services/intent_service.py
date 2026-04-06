"""
Intent Classification Service — intent_service.py
===================================================
Classifies user queries into one of four intents using gpt-4o-mini.
Drop into app/services/ alongside retriever_service.py.

Usage:
    service = IntentService()
    result  = service.classify("How does consistent hashing work?")
    print(result.label)       # IntentLabel.CONCEPT_EXPLANATION
    print(result.confidence)  # 0.95
    print(result.raw)         # "CONCEPT_EXPLANATION"
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Optional

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent label enum
# ---------------------------------------------------------------------------

class IntentLabel(str, Enum):
    CONCEPT_EXPLANATION   = "CONCEPT_EXPLANATION"
    SYSTEM_DESIGN_QUESTION = "SYSTEM_DESIGN_QUESTION"
    DESIGN_REVIEW         = "DESIGN_REVIEW"
    OUT_OF_SCOPE          = "OUT_OF_SCOPE"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """
    Output of a single classification call.

    Attributes:
        label:      The classified intent as an IntentLabel enum value.
        confidence: Model's self-reported confidence [0.0, 1.0].
                    Parsed from the JSON response; falls back to 0.0 if
                    the model omits it.
        raw:        The exact label string returned by the model before
                    parsing — useful for debugging unexpected outputs.
        query:      The original query that was classified.
    """
    label:      IntentLabel
    confidence: float
    raw:        str
    query:      str


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# Kept as a module-level constant so it can be version-controlled separately
# and swapped without touching service logic. In the full project this lives
# in app/prompts/intent_classifier.j2 — here it's inlined for portability.

INTENT_CLASSIFICATION_PROMPT = """\
You are an intent classifier for a system design learning assistant.
Your only job is to classify the user's query into exactly one of the \
following intents.

## Intent definitions

CONCEPT_EXPLANATION
    The user wants to understand a technical concept, pattern, or \
    technology. They are asking "what is", "how does", "explain", \
    "what does X mean", or similar.
    Examples:
      - "What is consistent hashing?"
      - "How does a bloom filter work?"
      - "Explain CAP theorem to me."
      - "What is the difference between SQL and NoSQL?"

SYSTEM_DESIGN_QUESTION
    The user wants to design a system from scratch or asks how to approach \
    building something at scale. They describe a product or feature and \
    want an architecture.
    Examples:
      - "Design a URL shortener like bit.ly."
      - "How would you build a notification service for 10 million users?"
      - "What's the architecture for a ride-sharing app?"
      - "How do I design a rate limiter?"

DESIGN_REVIEW
    The user has already described or produced a design and wants feedback, \
    critique, or validation. They share their own approach and ask if it is \
    correct, optimal, or how it could be improved.
    Examples:
      - "I'm using Redis for caching and Postgres for storage — is that good?"
      - "Here's my design for a chat system. What am I missing?"
      - "I chose Kafka over RabbitMQ for this use case. Does that make sense?"
      - "Review my database schema for a social feed."

OUT_OF_SCOPE
    The query is not related to software engineering, system design, or \
    computer science. This includes general knowledge, personal advice, \
    math unrelated to CS, or anything outside the assistant's domain.
    Examples:
      - "What's the weather in Mumbai?"
      - "Write me a poem."
      - "Who won the IPL last year?"
      - "Help me with my taxes."

## Output format

Respond with a JSON object and nothing else — no markdown, no explanation.
Use this exact schema:

{{
  "label":      "<one of the four intent names above>",
  "confidence": <float between 0.0 and 1.0>
}}

## Rules

1. Always return exactly one label. Never return multiple.
2. If the query is ambiguous between CONCEPT_EXPLANATION and \
SYSTEM_DESIGN_QUESTION, prefer SYSTEM_DESIGN_QUESTION when the user \
mentions scale, users, or building something.
3. If the query is ambiguous between SYSTEM_DESIGN_QUESTION and \
DESIGN_REVIEW, prefer DESIGN_REVIEW when the user shares their own \
existing design or asks "is this correct / good / right".
4. Default to OUT_OF_SCOPE only when there is no reasonable CS or \
system design interpretation.
5. Never explain your reasoning. Return only the JSON object.

## User query to classify

{query}
"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class IntentService:
    """
    Thin service that wraps the OpenAI chat API to classify query intent.

    Design choices:
    - Uses gpt-4o-mini: cheap, fast, and sufficient for classification.
      gpt-4o is overkill here; save it for generation.
    - temperature=0 for deterministic, reproducible output.
    - top_p=1, presence_penalty=0, frequency_penalty=0 — no sampling noise.
    - max_tokens=60: the response is always a small JSON object; capping
      tokens prevents runaway output and reduces latency.
    - JSON is parsed with a strict validator that raises on unknown labels,
      so classification errors surface immediately rather than silently
      producing bad intent values downstream.
    """

    MODEL          = "gpt-4o-mini"
    MAX_TOKENS     = 60
    TEMPERATURE    = 0
    FALLBACK_LABEL = IntentLabel.OUT_OF_SCOPE

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> IntentResult:
        """
        Classify a single query and return an IntentResult.

        Args:
            query: Raw user message string. Whitespace is stripped before
                   sending to the model.

        Returns:
            IntentResult with label, confidence, raw response, and original query.

        Raises:
            No exceptions are raised to callers — all errors are caught,
            logged, and returned as OUT_OF_SCOPE with confidence 0.0.
            This makes the classifier safe to call from the orchestrator
            without try/except boilerplate at the call site.
        """
        query = query.strip()
        if not query:
            logger.warning("classify() called with empty query — returning OUT_OF_SCOPE")
            return self._fallback(query, reason="empty query")

        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)

        try:
            raw_label, confidence = self._call_api(prompt)
            label = self._parse_label(raw_label)
            logger.info(
                "classify(%r) -> %s (confidence=%.2f)",
                query[:80], label.value, confidence,
            )
            return IntentResult(
                label=label,
                confidence=confidence,
                raw=raw_label,
                query=query,
            )

        except OpenAIError as exc:
            logger.error("OpenAI API error during classification: %s", exc)
            return self._fallback(query, reason=str(exc))

        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse classification response: %s", exc)
            return self._fallback(query, reason=str(exc))

    def classify_batch(self, queries: list[str]) -> list[IntentResult]:
        """
        Classify a list of queries sequentially.

        OpenAI does not support batch classification in a single chat call
        reliably — separate calls guarantee clean, unambiguous per-query
        outputs. For high-throughput ingestion, use the Batch API instead.

        Args:
            queries: List of raw query strings.

        Returns:
            List of IntentResult in the same order as input.
        """
        return [self.classify(q) for q in queries]

    async def classify_intent(self, query: str) -> str:
        """
        Async-friendly wrapper used by the orchestrator.

        Runs the synchronous `classify()` call in a thread-pool executor so
        it does not block the FastAPI event loop.

        Args:
            query: Raw user message string.

        Returns:
            The intent label as a plain string (e.g. "CONCEPT_EXPLANATION").
            Falls back to "OUT_OF_SCOPE" on any error.
        """
        loop = asyncio.get_event_loop()
        result: IntentResult = await loop.run_in_executor(
            None, partial(self.classify, query)
        )
        return result.label.value

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> tuple[str, float]:
        """
        Make the chat completion call and return (raw_label, confidence).

        The model is instructed to return only JSON. We extract the content,
        parse it, and pull out the two fields. Any deviation from the schema
        raises an exception caught by classify().
        """
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise intent classifier. "
                        "Always respond with valid JSON only. "
                        "No markdown. No explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0,
        )

        content = response.choices[0].message.content.strip()
        logger.debug("Raw model response: %r", content)

        # Strip markdown code fences if the model wraps output despite instructions
        if content.startswith("```"):
            content = content.strip("`").lstrip("json").strip()

        parsed = json.loads(content)

        raw_label  = str(parsed["label"]).strip().upper()
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        return raw_label, confidence

    def _parse_label(self, raw: str) -> IntentLabel:
        """
        Map the raw string label to an IntentLabel enum value.

        Raises ValueError for unrecognised labels so the error surfaces
        clearly rather than silently falling through.
        """
        try:
            return IntentLabel(raw)
        except ValueError:
            valid = [l.value for l in IntentLabel]
            raise ValueError(
                f"Model returned unknown label {raw!r}. Valid labels: {valid}"
            )

    def _fallback(self, query: str, reason: str = "") -> IntentResult:
        """Return a safe OUT_OF_SCOPE result when something goes wrong."""
        logger.warning("Falling back to OUT_OF_SCOPE. Reason: %s", reason)
        return IntentResult(
            label=IntentLabel.OUT_OF_SCOPE,
            confidence=0.0,
            raw="OUT_OF_SCOPE",
            query=query,
        )


# ---------------------------------------------------------------------------
# Smoke test — runs only when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    service = IntentService()

    test_cases = [
        # (query, expected_label)
        ("What is consistent hashing?",                        IntentLabel.CONCEPT_EXPLANATION),
        ("Explain the CAP theorem.",                           IntentLabel.CONCEPT_EXPLANATION),
        ("Design a URL shortener for 1 billion users.",        IntentLabel.SYSTEM_DESIGN_QUESTION),
        ("How would you build a notification service?",        IntentLabel.SYSTEM_DESIGN_QUESTION),
        ("I'm using Redis for sessions — is that a good idea?", IntentLabel.DESIGN_REVIEW),
        ("Here's my chat app design. What am I missing?",      IntentLabel.DESIGN_REVIEW),
        ("What's the weather in Mumbai?",                      IntentLabel.OUT_OF_SCOPE),
        ("Write me a poem about databases.",                   IntentLabel.OUT_OF_SCOPE),
    ]

    passed = 0
    print(f"\n{'Query':<55} {'Expected':<28} {'Got':<28} {'Conf':>6}  {'OK?'}")
    print("-" * 130)

    for query, expected in test_cases:
        result = service.classify(query)
        ok = result.label == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(
            f"{query[:54]:<55} {expected.value:<28} {result.label.value:<28} "
            f"{result.confidence:>6.2f}  {status}"
        )

    print(f"\n{passed}/{len(test_cases)} tests passed.")


# Singleton instance for use across routes/services
intent_service = IntentService()