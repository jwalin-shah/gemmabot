"""Safety Agent — monitors scene + plan for hazards and constraint violations."""

from __future__ import annotations

from src.client import CerebrasClient, InferenceResult

SYSTEM_PROMPT = """You are a robot safety monitor. You review scene descriptions and proposed
action plans to identify risks.

Return a JSON object with:
- "safe": boolean — whether the plan can proceed
- "risk_level": "low" | "medium" | "high"
- "issues": list of specific safety concerns found
- "recommendation": what to do instead (if unsafe)
- "severity": 0-1 score

Be conservative — flag anything ambiguous as a risk."""


class SafetyAgent:
    """Reviews plans for safety before execution."""

    def __init__(self, client: CerebrasClient) -> None:
        self._client = client

    def review(
        self,
        scene_analysis: str,
        action_plan: str,
    ) -> InferenceResult:
        """Review a scene + action plan for safety."""
        prompt = (
            f"Scene analysis:\n{scene_analysis}\n\n"
            f"Proposed action plan:\n{action_plan}"
        )
        return self._client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a robot safety monitor. Review the scene and plan. "
                        "Return JSON with fields: safe (bool), risk_level, issues (list), "
                        "recommendation, severity (0-1)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )