"""Branch Registry — factory for creating and managing specialist branch agents."""

from __future__ import annotations

import time
from typing import Any

from src.agents import VisionAgent, ActionAgent, SafetyAgent
from src.client import CerebrasClient
from src.command_center.types import Branch, BranchOutput


class BranchRegistry:
    """Holds all specialist agents that the Command Center root can delegate to."""

    def __init__(self, client: CerebrasClient) -> None:
        self._client = client
        self._branches: dict[Branch, Any] = {}

    def register_all(self) -> None:
        """Register all available branch agents."""
        self._branches = {
            Branch.VISION: VisionAgent(self._client),
            Branch.ACTION_PLANNER: ActionAgent(self._client),
            Branch.SAFETY: SafetyAgent(self._client),
            Branch.SUMMARIZER: self._make_summarizer(),
            Branch.ORACLE: self._make_oracle(),
        }

    def get(self, branch: Branch) -> Any:
        """Get a branch agent by enum."""
        if branch not in self._branches:
            msg = f"Branch '{branch.value}' not registered"
            raise KeyError(msg)
        return self._branches[branch]

    def run_branch(
        self,
        branch: Branch,
        prompt: str,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> BranchOutput:
        """Execute a branch agent and return structured output with timing."""
        agent = self.get(branch)
        t0 = time.perf_counter()

        try:
            if branch == Branch.VISION:
                image_b64 = (context or {}).get("image_b64", "")
                result = agent.analyze(image_b64, prompt=prompt)
            elif branch == Branch.ACTION_PLANNER:
                scene = (context or {}).get("scene_analysis", "")
                result = agent.plan(scene, task=prompt)
            elif branch == Branch.SAFETY:
                scene = (context or {}).get("scene_analysis", "")
                plan = (context or {}).get("action_plan", "")
                result = agent.review(scene, plan)
            else:
                result = self._client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt=system_prompt,
                )

            elapsed = (time.perf_counter() - t0) * 1000

            # Try to parse structured JSON from response
            structured: dict[str, Any] = {}
            content = result.content.strip()
            if content.startswith("{"):
                import json
                try:
                    structured = json.loads(content)
                except json.JSONDecodeError:
                    pass

            return BranchOutput(
                branch=branch,
                content=content,
                structured=structured,
                latency_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            return BranchOutput(
                branch=branch,
                content="",
                error=str(e),
                latency_ms=elapsed,
            )

    def run_parallel(
        self,
        branches: list[Branch],
        prompts: dict[Branch, str],
        context: dict[str, Any] | None = None,
        system_prompts: dict[Branch, str] | None = None,
    ) -> list[BranchOutput]:
        """Run multiple branches in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        system_prompts = system_prompts or {}

        def _run(b: Branch) -> BranchOutput:
            return self.run_branch(
                b,
                prompts.get(b, ""),
                context=context,
                system_prompt=system_prompts.get(b),
            )

        with ThreadPoolExecutor(max_workers=len(branches)) as ex:
            outputs = list(ex.map(_run, branches))

        return outputs

    # ------------------------------------------------------------------
    # Inline branch constructors
    # ------------------------------------------------------------------

    def _make_summarizer(self) -> _GeneralAgent:
        return _GeneralAgent(
            self._client,
            "You are a summarizer. Condense complex information into concise structured summaries.",
        )

    def _make_oracle(self) -> _GeneralAgent:
        return _GeneralAgent(
            self._client,
            "You are a general reasoning agent. Answer questions and provide analysis.",
        )


class _GeneralAgent:
    """Wrapper for non-specialist agents that just use text prompts."""

    def __init__(self, client: CerebrasClient, system_prompt: str) -> None:
        self._client = client
        self._system_prompt = system_prompt

    def __call__(self, prompt: str) -> str:
        result = self._client.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=self._system_prompt,
        )
        return result.content
