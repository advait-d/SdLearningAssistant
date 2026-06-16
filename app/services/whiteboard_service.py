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

WHITEBOARD_GRADE_PROMPT = """You are an expert system design interviewer reviewing a candidate's architecture diagram.

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

Your task: Review this diagram as an engaged interviewer would. Evaluate whether the diagram adequately addresses the problem.

Respond ONLY with valid JSON in this exact shape (no markdown, no prose):
{{
  "feedback": "<2-3 sentence natural interviewer reaction — acknowledge what they drew, what's missing, transition to your follow-up. Sound like a real tech interviewer, not a report.>",
  "follow_up": "<single sharp follow-up question based on the diagram — probe for something missing or a tradeoff>",
  "annotations": {{
    "correct": ["<component or concept the candidate correctly included>", ...],
    "missing": ["<critical missing component or concept>", ...],
    "suggestions": ["<specific improvement or detail to add>", ...]
  }},
  "diagram_score": <integer 1-10>,
  "diagram_verdict": "<EXCELLENT|GOOD|INCOMPLETE|MISSING_CRITICAL_COMPONENTS>"
}}

Scoring guide:
- 8-10: All critical components present, proper connections, handles scale
- 6-7: Core components present but missing resilience/scale considerations
- 4-5: Has the right idea but missing 2+ critical components
- 1-3: Very basic, missing most architectural elements

For system_design problems, critical components typically include: Load balancer, CDN, API gateway, primary database, cache layer, message queue (if async), monitoring/alerting.
For low_level_design, critical components include: Core domain entities, key interfaces/contracts, data persistence layer, key design patterns applied.
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
        annotations = raw.get("annotations", {})
        return {
            "feedback": raw.get(
                "feedback",
                f"I can see you've started drawing your architecture for '{prompt}'. Let me give you some feedback."
            ),
            "follow_up": raw.get(
                "follow_up",
                "Can you walk me through how data flows through the components you've drawn?"
            ),
            "annotations": {
                "correct": annotations.get("correct", []),
                "missing": annotations.get("missing", []),
                "suggestions": annotations.get("suggestions", []),
            },
            "diagram_score": raw.get("diagram_score", 5),
            "diagram_verdict": raw.get("diagram_verdict", "INCOMPLETE"),
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
            "follow_up": "Can you explain the data flow between the components in your diagram?",
            "annotations": {
                "correct": labels[:3] if has_labels else [],
                "missing": (
                    ["Load balancer", "Database", "Cache layer"]
                    if topic == "system_design"
                    else ["Core entities", "Data persistence", "Key interfaces"]
                ),
                "suggestions": ["Add connection arrows to show data flow", "Label each component clearly"],
            },
            "diagram_score": 4,
            "diagram_verdict": "INCOMPLETE",
        }


# ─── Singleton ────────────────────────────────────────────────────────────────

whiteboard_grading_service = WhiteboardGradingService()
