"""
Interview Service — app/services/interview_service.py
=====================================================
Manages AI-powered mock interview sessions with topic-aware prompting,
multi-turn conversation management, and structured scorecard generation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Interview Configuration ──────────────────────────────────────────────────

class InterviewTopic(str, Enum):
    SYSTEM_DESIGN = "system_design"
    LLD = "low_level_design"
    BEHAVIORAL = "behavioral"
    ML_DESIGN = "ml_design"
    DSA = "dsa"


class InterviewDifficulty(str, Enum):
    L3_L4 = "junior"       # SDE-1 / SDE-2
    L5 = "mid_senior"      # Senior
    L6 = "staff"           # Staff
    L7 = "principal"       # Principal


# ─── System Prompts ───────────────────────────────────────────────────────────

INTERVIEWER_PROMPTS: Dict[InterviewTopic, str] = {
    InterviewTopic.SYSTEM_DESIGN: """You are Alex, a Staff Engineer at a top-tier tech company conducting a system design interview.

RULES — follow these strictly:
1. Ask ONE clarifying question or follow-up at a time. Never dump multiple questions.
2. Stay strictly in character as a tough but fair interviewer.
3. Probe for: scale estimates, data modeling choices, tradeoffs, failure modes, bottlenecks.
4. Push back on hand-wavy answers. Ask "why" and "what happens if this fails?"
5. Guide the candidate when they're stuck — give a small hint, not the answer.
6. After about 10-12 exchanges, naturally wrap up the interview.
7. Keep responses under 100 words. Be concise like a real interviewer.
8. Never break character. Never say you're an AI.

The candidate is designing: "{prompt}"
Difficulty level: {difficulty}

Start by briefly introducing yourself and asking the candidate to begin with requirements.""",

    InterviewTopic.LLD: """You are Priya, a Senior Engineer conducting a low-level design / object-oriented design interview.

RULES:
1. Ask ONE question at a time about class design, patterns, or edge cases.
2. Probe for: SOLID principles, design patterns, concurrency, error handling.
3. Ask the candidate to walk through their class hierarchy and key methods.
4. Push for concrete method signatures and data structures.
5. Challenge assumptions about thread safety and scalability.
6. Keep responses under 100 words.
7. Never break character.

The candidate is designing: "{prompt}"
Difficulty level: {difficulty}

Start by asking the candidate to identify the core entities and their relationships.""",

    InterviewTopic.BEHAVIORAL: """You are Jordan, a Senior Engineering Manager conducting a behavioral interview.

RULES:
1. Ask ONE behavioral question at a time. Focus on the STAR framework.
2. If the candidate's answer lacks clear Situation, Task, Action, or Result, you MUST explicitly coach them. For example: "I hear the result, but what was your specific action?" or "Can you frame that using the STAR method for me? What was the Situation?"
3. Push for "I" not "We". If they say "We built X", ask "What was your specific individual contribution?"
4. Ask structured follow-ups based on their STAR response: 
   - After "Result", ask "What would you do differently next time?" (Learn/Grow)
   - After "Action", ask "Did you consider any alternative approaches?"
5. Keep responses under 80 words. Be conversational but authoritative.
6. Never break character.

Focus area: {prompt}
Difficulty level: {difficulty}

Start by outlining that you'll be looking for structured answers using the STAR method, then ask your first question related to the focus area.""",

    InterviewTopic.ML_DESIGN: """You are Dr. Chen, a Principal ML Engineer conducting an ML system design interview.

RULES:
1. Ask ONE question at a time about model architecture, data pipeline, or serving.
2. Probe for: feature engineering choices, model selection rationale, evaluation metrics.
3. Push on production concerns: latency, throughput, data drift, A/B testing.
4. Challenge offline vs online metric alignment.
5. Ask about failure modes and monitoring.
6. Keep responses under 100 words.
7. Never break character.

The candidate is designing: "{prompt}"
Difficulty level: {difficulty}

Start by asking the candidate to clarify the problem and define success metrics.""",

    InterviewTopic.DSA: """You are Sam, a Senior Engineer conducting a coding/algorithms interview.

RULES:
1. Present ONE problem clearly with examples.
2. Let the candidate think aloud before coding.
3. Ask about time/space complexity after they propose an approach.
4. If stuck, give a small hint — not the solution.
5. Ask about edge cases: empty input, duplicates, overflow.
6. Challenge them to optimize after a brute-force solution.
7. Keep responses under 100 words.
8. Never break character.

Problem area: {prompt}
Difficulty level: {difficulty}

Start by presenting the problem clearly with 1-2 examples.""",
}


# Scorecard evaluation prompt
SCORECARD_PROMPT = """Analyze this interview transcript and produce a structured evaluation.

Interview Type: {topic}
Problem: {prompt}
Difficulty: {difficulty}

Transcript:
\"\"\"
{transcript}
\"\"\"

Respond ONLY with valid JSON in this exact shape (no markdown, no prose, no code fences):
{{
  "overall_score": <1-10>,
  "hire_decision": "<STRONG_HIRE|HIRE|LEAN_HIRE|LEAN_NO_HIRE|NO_HIRE>",
  "dimensions": [
    {{"name": "<dimension>", "score": <1-10>, "max": 10, "feedback": "<specific 15-20 word feedback>"}},
    ...
  ],
  "strengths": ["<specific strength 1>", "<specific strength 2>"],
  "improvements": ["<specific improvement 1>", "<specific improvement 2>", "<specific improvement 3>"],
  "summary": "<3-4 sentence overall assessment with actionable advice>"
}}

Use these dimensions based on interview type:
- system_design: Requirements Clarification, High-Level Architecture, Data Modeling, Scalability & Performance, Tradeoff Analysis, Communication
- low_level_design: OOP Design, Design Patterns, API Design, Concurrency/Thread Safety, Edge Cases, Code Quality
- behavioral: STAR Structure, Leadership Signal, Self-Awareness, Impact & Metrics, Communication
- ml_design: Problem Formulation, Feature Engineering, Model Selection, Evaluation Strategy, Production Readiness, Communication
- dsa: Problem Understanding, Algorithm Design, Complexity Analysis, Code Quality, Edge Cases, Communication"""


# ─── Session Management ──────────────────────────────────────────────────────

@dataclass
class InterviewSession:
    """In-memory interview session (stateless per-request in production)."""
    session_id: str
    topic: InterviewTopic
    prompt: str
    difficulty: InterviewDifficulty
    messages: List[Dict[str, str]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    turn_count: int = 0
    is_ended: bool = False


# Simple in-memory store — replace with Redis/DB for multi-instance production
_sessions: Dict[str, InterviewSession] = {}


class InterviewService:
    """Orchestrates multi-turn AI interviews with structured evaluation."""

    def __init__(self):
        from app.services.llm_service import LLMService
        self._llm = LLMService()

    async def start_session(
        self,
        session_id: str,
        topic: InterviewTopic,
        prompt: str,
        difficulty: InterviewDifficulty = InterviewDifficulty.L5,
        company: str = "general",
        context: Optional[str] = None,
    ) -> Dict:
        """Start a new interview session and return the interviewer's opening."""

        system_prompt = INTERVIEWER_PROMPTS[topic].format(
            prompt=prompt,
            difficulty=difficulty.value,
        )

        if company != "general":
            system_prompt += f"\n\nCompany Context: You are interviewing for a role at {company.title()}. Adjust your questions, priorities, and evaluation criteria to strongly align with {company.title()}'s engineering culture, leadership principles, and architectural scale."

        # Optionally inject RAG context from chapters
        if context:
            system_prompt += f"\n\nRelevant technical context for your reference (use to probe deeper):\n{context[:3000]}"

        messages = [{"role": "system", "content": system_prompt}]

        # Get the interviewer's opening
        try:
            opening = await self._llm.generate(messages)
        except Exception as e:
            logger.error("Failed to generate interview opening: %s", e)
            opening = self._fallback_opening(topic, prompt)

        messages.append({"role": "assistant", "content": opening})

        session = InterviewSession(
            session_id=session_id,
            topic=topic,
            prompt=prompt,
            difficulty=difficulty,
            messages=messages,
            turn_count=1,
        )
        _sessions[session_id] = session

        logger.info("Interview session started: %s [%s] %s", session_id, topic.value, prompt)

        return {
            "session_id": session_id,
            "response": opening,
            "turn": 1,
            "topic": topic.value,
            "prompt": prompt,
        }

    async def process_turn(self, session_id: str, user_message: str) -> Dict:
        """Process a candidate's response and return the interviewer's follow-up."""

        session = _sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "session_id": session_id}

        if session.is_ended:
            return {"error": "Interview has already ended", "session_id": session_id}

        # Add user message
        session.messages.append({"role": "user", "content": user_message})
        session.turn_count += 1

        # Check if we should wrap up naturally
        should_wrap_up = session.turn_count >= 20  # ~10 exchanges

        if should_wrap_up:
            wrap_instruction = {
                "role": "system",
                "content": "The interview is nearing the end. Wrap up naturally within 1-2 more exchanges. Thank the candidate."
            }
            messages_with_instruction = session.messages + [wrap_instruction]
        else:
            messages_with_instruction = session.messages

        # Generate interviewer response
        try:
            response = await self._llm.generate(messages_with_instruction)
        except Exception as e:
            logger.error("Failed to generate interview turn: %s", e)
            response = "That's an interesting point. Could you elaborate on the tradeoffs you'd consider?"

        session.messages.append({"role": "assistant", "content": response})

        return {
            "session_id": session_id,
            "response": response,
            "turn": session.turn_count,
            "should_end_soon": should_wrap_up,
        }

    async def end_session(self, session_id: str) -> Dict:
        """End the interview and generate a structured scorecard."""

        session = _sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "session_id": session_id}

        session.is_ended = True

        # Build transcript
        transcript = self._build_transcript(session)

        # Generate scorecard
        scorecard_prompt = SCORECARD_PROMPT.format(
            topic=session.topic.value,
            prompt=session.prompt,
            difficulty=session.difficulty.value,
            transcript=transcript,
        )

        try:
            raw = await self._llm.generate([
                {"role": "system", "content": "You are an expert interview evaluator. Respond only with valid JSON."},
                {"role": "user", "content": scorecard_prompt},
            ])

            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                import json
                scorecard = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            logger.error("Scorecard generation failed: %s", e)
            scorecard = self._fallback_scorecard(session)

        # Calculate duration
        duration_seconds = int(time.time() - session.started_at)

        result = {
            "session_id": session_id,
            "scorecard": scorecard,
            "transcript": transcript,
            "turns": session.turn_count,
            "duration_seconds": duration_seconds,
            "topic": session.topic.value,
            "prompt": session.prompt,
        }

        # Cleanup session from memory (keep last 100)
        if len(_sessions) > 100:
            oldest_key = min(_sessions, key=lambda k: _sessions[k].started_at)
            del _sessions[oldest_key]

        return result

    def _build_transcript(self, session: InterviewSession) -> str:
        """Build a readable transcript from session messages."""
        lines = []
        for msg in session.messages:
            if msg["role"] == "system":
                continue
            role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)

    def _fallback_opening(self, topic: InterviewTopic, prompt: str) -> str:
        """Fallback opening if LLM fails."""
        openers = {
            InterviewTopic.SYSTEM_DESIGN: f"Hi, I'm Alex. Today we'll be designing {prompt}. Why don't you start by walking me through the key requirements you'd clarify with a product manager?",
            InterviewTopic.LLD: f"Hi, I'm Priya. For today's session, we'll work through the low-level design of {prompt}. Let's start — what are the core entities you'd identify?",
            InterviewTopic.BEHAVIORAL: f"Hi, I'm Jordan. I'm looking forward to our conversation today. Before we dive into specifics, tell me about a recent project you're proud of.",
            InterviewTopic.ML_DESIGN: f"Hi, I'm Dr. Chen. Today we're designing {prompt}. Let's start with the basics — how would you formulate this as an ML problem, and what would success look like?",
            InterviewTopic.DSA: f"Hi, I'm Sam. Let's work through a problem related to {prompt}. I'll present the problem, and then we can discuss your approach before you start coding.",
        }
        return openers.get(topic, f"Let's begin discussing {prompt}. Please start with your initial thoughts.")

    def _fallback_scorecard(self, session: InterviewSession) -> Dict:
        """Fallback scorecard if LLM evaluation fails."""
        return {
            "overall_score": 6,
            "hire_decision": "LEAN_HIRE",
            "dimensions": [
                {"name": "Communication", "score": 7, "max": 10, "feedback": "Clear and structured explanation throughout the discussion."},
                {"name": "Technical Depth", "score": 6, "max": 10, "feedback": "Good fundamentals shown but some areas lacked depth."},
                {"name": "Problem Solving", "score": 6, "max": 10, "feedback": "Reasonable approach but missed some edge cases."},
            ],
            "strengths": ["Clear communication", "Good high-level structure"],
            "improvements": ["Go deeper on tradeoffs", "Quantify with numbers", "Consider failure modes explicitly"],
            "summary": f"A solid interview performance on {session.prompt}. The candidate demonstrated good fundamentals and clear communication. Focus on deeper tradeoff analysis and failure mode discussion to reach the next level.",
        }


# ─── Singleton ─────────────────────────────────────────────────────────────────

interview_service = InterviewService()
