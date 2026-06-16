"""
Whiteboard Grading Service — app/services/whiteboard_service.py
===============================================================
Grades architecture diagrams drawn during a mock interview session.

Unlike the generic whiteboard grader, this service:
  - Uses the live interview session context (prompt + conversation excerpt)
  - Returns structured annotations (missing/correct/suggestions) for UI overlay
  - Generates a natural follow-up question the AI interviewer would ask
  - Integrates into the interview conversation as a real interviewer would
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


# ─── Grading Prompt ──────────────────────────────────────────────────────────

WHITEBOARD_GRADE_PROMPT = """You are an expert Staff/Principal system design interviewer reviewing a candidate's architecture diagram.

Interview Context:
- Problem: {prompt}
- Interview Type: {topic}
- Difficulty: {difficulty}

What the candidate has drawn (extracted labels and component count):
- Text labels on diagram: [{labels}]
- Total shapes/components: {component_count}
- Connection/arrow count: {connection_count}

Recent conversation context:
{conversation_context}

Your task: Review this diagram as a Staff-level interviewer would. Evaluate whether the diagram adequately addresses the problem, focusing on scale, reliability, cost, and proper component separation.

Respond ONLY with valid JSON in this exact shape (no markdown, no prose):
{{
  "feedback": "<2-3 sentence natural interviewer reaction — acknowledge what they drew, what's missing, transition to your follow-up. Sound like a real tech interviewer.>",
  "hiring_recommendation": "<Strong Hire|Hire|Leaning Hire|No Hire>",
  "architecture_score": <integer 1-100>,
  "dimensions": {{
    "coverage": <0-100>,
    "scalability": <0-100>,
    "reliability": <0-100>,
    "cost": <0-100>,
    "security": <0-100>,
    "observability": <0-100>,
    "tradeoff_quality": <0-100>
  }},
  "issues": {{
    "critical": [
      {{ "text": "<Issue description>", "node_label": "<The exact text label from the diagram causing the issue, or null if missing>" }}
    ],
    "important": [
      {{ "text": "<Issue description>", "node_label": "<The exact text label from the diagram causing the issue, or null>" }}
    ],
    "nice_to_have": [
      {{ "text": "<Issue description>", "node_label": "<The exact text label from the diagram causing the issue, or null>" }}
    ]
  }},
  "follow_up_questions": [
    "<sharp follow-up question 1>",
    "<sharp follow-up question 2>",
    "<sharp follow-up question 3>"
  ]
}}

Scoring guide:
- 90-100: All critical components present, handles scale gracefully, proper tradeoffs.
- 70-89: Solid core, missing some resilience/scale considerations.
- 50-69: Has the right idea but missing multiple critical components.
- <50: Very basic, fundamentally flawed or incomplete.
"""


# ─── Service ─────────────────────────────────────────────────────────────────

class WhiteboardGradingService:
    """
    Context-aware whiteboard grader that integrates with the interview loop.
    Uses the interview session context to produce relevant, personalized feedback.
    """

    async def grade_diagram(
        self,
        prompt: str,
        topic: str,
        difficulty: str,
        elements: List[Dict[str, Any]],
        conversation_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Grade an Excalidraw diagram in the context of an ongoing interview.

        Args:
            prompt: The interview problem (e.g. "Design Twitter's feed system")
            topic: Interview topic (e.g. "system_design")
            difficulty: Difficulty level (e.g. "mid_senior")
            elements: Raw Excalidraw elements list from the frontend
            conversation_messages: Recent interview messages for context

        Returns:
            Dict with: feedback, follow_up, annotations, diagram_score, diagram_verdict
        """
        # Extract meaningful content from Excalidraw elements
        text_labels = self._extract_text_labels(elements)
        component_count = len([e for e in elements if e.get("type") != "text"])
        connection_count = len([e for e in elements if e.get("type") == "arrow"])

        # Build conversation context snippet (last 4 messages)
        conversation_context = self._build_conversation_context(
            conversation_messages or [], max_messages=4
        )

        grading_prompt = WHITEBOARD_GRADE_PROMPT.format(
            prompt=prompt,
            topic=topic,
            difficulty=difficulty,
            labels=", ".join(text_labels) if text_labels else "no text labels",
            component_count=component_count,
            connection_count=connection_count,
            conversation_context=conversation_context or "No prior conversation.",
        )

        try:
            raw = await llm_service.generate_structured_response(
                system_prompt=(
                    "You are an expert technical interviewer reviewing architecture diagrams. "
                    "Respond only with valid JSON — no markdown, no code fences."
                ),
                user_input=grading_prompt,
                provider="gemini",
                model="gemini-2.5-flash",
                temperature=0.4,
            )
            # Validate required keys
            result = self._validate_and_normalize(raw, prompt, text_labels)
            return result

        except Exception as e:
            logger.error("Whiteboard grading failed: %s", e)
            return self._fallback_result(prompt, text_labels, topic)

    def _extract_text_labels(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Extract all non-empty text content from Excalidraw elements."""
        labels = []
        for el in elements:
            text = el.get("text", "").strip()
            if text and len(text) > 0:
                labels.append(text)
            # Also check label field for shapes with embedded text
            label = el.get("label", {})
            if isinstance(label, dict):
                label_text = label.get("text", "").strip()
                if label_text:
                    labels.append(label_text)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for l in labels:
            if l not in seen:
                seen.add(l)
                unique.append(l)
        return unique[:30]  # Cap at 30 labels to avoid prompt bloat

    def _build_conversation_context(
        self, messages: List[Dict[str, str]], max_messages: int = 4
    ) -> str:
        """Format recent conversation messages for the grading prompt."""
        # Skip system messages, take last N user/assistant messages
        non_system = [m for m in messages if m.get("role") in ("user", "assistant")]
        recent = non_system[-max_messages:] if len(non_system) > max_messages else non_system
        lines = []
        for m in recent:
            role = "Interviewer" if m["role"] == "assistant" else "Candidate"
            content = m["content"][:300]  # Truncate long messages
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _validate_and_normalize(
        self,
        raw: Dict[str, Any],
        prompt: str,
        text_labels: List[str],
    ) -> Dict[str, Any]:
        """Ensure all required keys exist with sensible defaults."""
        issues = raw.get("issues", {})
        dimensions = raw.get("dimensions", {})
        return {
            "feedback": raw.get(
                "feedback",
                f"I can see you've started drawing your architecture for '{prompt}'. Let me give you some feedback."
            ),
            "hiring_recommendation": raw.get("hiring_recommendation", "Leaning Hire"),
            "architecture_score": raw.get("architecture_score", 50),
            "dimensions": {
                "coverage": dimensions.get("coverage", 50),
                "scalability": dimensions.get("scalability", 50),
                "reliability": dimensions.get("reliability", 50),
                "cost": dimensions.get("cost", 50),
                "security": dimensions.get("security", 50),
                "observability": dimensions.get("observability", 50),
                "tradeoff_quality": dimensions.get("tradeoff_quality", 50),
            },
            "issues": {
                "critical": issues.get("critical", []),
                "important": issues.get("important", []),
                "nice_to_have": issues.get("nice_to_have", []),
            },
            "follow_up_questions": raw.get(
                "follow_up_questions",
                ["Can you walk me through how data flows through the components you've drawn?"]
            )
        }

    def _fallback_result(
        self, prompt: str, labels: List[str], topic: str
    ) -> Dict[str, Any]:
        """Return a sensible fallback when LLM grading fails."""
        has_labels = len(labels) > 0
        drawn_desc = f"added {len(labels)} components" if has_labels else "started drawing"
        return {
            "feedback": (
                f"I can see you've {drawn_desc} "
                f"for the {prompt} problem. "
                "I couldn't analyze the diagram fully right now, but can you walk me through what you've drawn so far?"
            ),
            "hiring_recommendation": "Leaning Hire",
            "architecture_score": 40,
            "dimensions": {
                "coverage": 40, "scalability": 40, "reliability": 40, 
                "cost": 50, "security": 40, "observability": 20, "tradeoff_quality": 40
            },
            "issues": {
                "critical": [{"text": "Missing key components", "node_label": None}],
                "important": [{"text": "Explain data flow", "node_label": None}],
                "nice_to_have": []
            },
            "follow_up_questions": [
                "Can you explain the data flow between the components in your diagram?",
                "How does this system handle a sudden spike in traffic?",
                "Where are the single points of failure?"
            ]
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

whiteboard_grading_service = WhiteboardGradingService()
