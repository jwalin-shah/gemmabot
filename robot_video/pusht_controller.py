"""Hybrid PushT controller: systematic search + Gemma-guided push + adaptive feedback.

Phase 1 — Systematic Grid Search (no LLM):
    Sweep the pusher through a grid of positions covering the full 512x512 space.
    At each position, step the environment and check reward > 0 (contact = touching
    the T-block). When contact is detected, immediately transition to Phase 2.

Phase 2 — Gemma-Guided Pushing:
    Render the current frame (with Zone A-F grid overlay downsampled to 96x96).
    Ask Gemma: "You are touching the T-block. Look at the image. Which direction
    should you push to move it toward the green target?"  Gemma outputs XY target
    position via structured JSON schema.

Phase 3 — Adaptive Feedback:
    After each push, compare reward delta. Feed back to Gemma:
    "Last push increased/decreased reward by X." If reward increased, continue in
    a similar direction; if it decreased or stayed, try a different direction.

Usage:
    from robot_video.pusht_controller import HybridPushtController
    ctrl = HybridPushtController(api_key=...)
    ctrl.run_episode(max_steps=40)
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any

import gym_pusht.envs  # noqa: F401
import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from src.client import CerebrasClient
from src.config import CEREBRAS_API_KEY


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPACE_SIZE = 512.0
GRID_STEP = 64  # spacing between grid search points
GRID_OFFSET = GRID_STEP // 2  # start 32px in to avoid edges

# Gemma structured-output schema for Phase 2
PUSH_SCHEMA: dict[str, Any] = {
    "name": "push_target",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "target_x": {
                "type": "number",
                "description": "X position to move the pusher to (0-512) to push the T-block toward the green target",
            },
            "target_y": {
                "type": "number",
                "description": "Y position to move the pusher to (0-512) to push the T-block toward the green target",
            },
            "reasoning": {
                "type": "string",
                "description": "Where is the T-block relative to the target? Why push this way? (max 100 chars)",
            },
        },
        "required": ["target_x", "target_y", "reasoning"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class Phase:
    SEARCH = "search"
    CONTACT = "contact"
    PUSH = "push"
    DONE = "done"


# ---------------------------------------------------------------------------
# Data carrier for one step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    image_b64: str  # data URI of rendered frame
    reward: float
    pos: list[float]  # [x, y] pusher position
    phase: str
    message: str  # human-readable status
    step: int
    reward_delta: float = 0.0  # change in reward since last step
    gemma_reasoning: str = ""
    gemma_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Grid generator
# ---------------------------------------------------------------------------

def _generate_grid() -> list[list[float]]:
    """Generate a zigzag grid of (x, y) positions covering [0, 512]."""
    positions: list[list[float]] = []
    xs = list(range(GRID_OFFSET, int(SPACE_SIZE), GRID_STEP))
    ys = list(range(GRID_OFFSET, int(SPACE_SIZE), GRID_STEP))
    for i, y in enumerate(ys):
        row = xs if i % 2 == 0 else reversed(xs)
        for x in row:
            positions.append([float(x), float(y)])
    return positions


# ---------------------------------------------------------------------------
# Resize + grid overlay helper
# ---------------------------------------------------------------------------

def _render_with_grid(frame: np.ndarray) -> Image.Image:
    """Overlay a Zone A-F grid pattern at full 680x680 resolution for Gemma."""
    pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil)

    # 3x3 grid overlay for spatial reasoning — labels A B C / D E F / G H I
    w, h = pil.size
    cols, rows = 3, 3
    cell_w, cell_h = w / cols, h / rows

    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
            draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 100), width=1)
            label_idx = r * cols + c
            label = chr(ord("A") + label_idx)
            draw.text((x0 + 3, y0 + 2), label, fill=(0, 255, 100))

    return pil


def _frame_to_uri(pil_img: Image.Image, quality: int = 80) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------

class HybridPushtController:
    """Three-phase PushT controller combining scripted search with LLM guidance.

    Public API:
        run_episode(max_steps=40) -> list[StepResult]  # full episode trace
        step() -> StepResult                            # single step (caller loops)
    """

    def __init__(
        self,
        api_key: str | None = None,
        headless: bool = False,
    ) -> None:
        self._api_key = api_key or CEREBRAS_API_KEY
        self._client = CerebrasClient() if self._api_key else None
        self._headless = headless

        # Environment
        self._env: gym.Env | None = None
        self._step_count = 0

        # State
        self.phase: str = Phase.SEARCH
        self.pusher_pos: list[float] = [256.0, 256.0]
        self.last_reward: float = 0.0
        self.best_reward: float = 0.0
        self.last_action_was_good: bool | None = None  # True/False/None

        # Grid search state
        self._grid = _generate_grid()
        self._grid_idx = 0

        # Feedback history (for Phase 3)
        self._push_history: list[dict[str, Any]] = []

    # ---- Public API -------------------------------------------------------

    def reset(self) -> None:
        """Reset the environment and internal state for a new episode."""
        if self._env:
            self._env.close()
        self._env = gym.make(
            "gym_pusht/PushT-v0", render_mode="rgb_array", obs_type="pixels"
        )
        self._env.reset()
        self._step_count = 0
        self.phase = Phase.SEARCH
        self.pusher_pos = [256.0, 256.0]
        self.last_reward = 0.0
        self.best_reward = 0.0
        self.last_action_was_good = None
        self._grid_idx = 0
        self._push_history = []

    @property
    def env(self):
        if self._env is None:
            self.reset()
        return self._env

    def step(self) -> StepResult:
        """Execute one control step based on the current phase.

        Returns the rendered frame, reward, phase, and status message.
        The caller should inspect .phase to know what happened.
        """
        if self.phase == Phase.SEARCH:
            return self._do_search_step()
        elif self.phase == Phase.CONTACT:
            return self._do_contact_step()
        elif self.phase == Phase.PUSH:
            return self._do_push_step()
        elif self.phase == Phase.DONE:
            return self._do_idle_step()
        else:
            # fallback — shouldn't happen
            return self._do_search_step()

    def run_episode(self, max_steps: int = 40) -> list[StepResult]:
        """Run a full episode, collecting every step result.

        Returns a list of StepResult objects the caller can inspect or replay.
        """
        self.reset()
        results: list[StepResult] = []
        for _ in range(max_steps):
            res = self.step()
            results.append(res)
            if res.phase == Phase.DONE:
                break
        return results

    # ---- Phase implementations --------------------------------------------

    def _do_search_step(self) -> StepResult:
        """Phase 1: Move the pusher through the grid until contact is made."""
        e = self.env

        # Pick the next grid position
        if self._grid_idx < len(self._grid):
            pos = self._grid[self._grid_idx]
            self._grid_idx += 1
        else:
            # Grid exhausted — loop back from start (shouldn't normally happen)
            self._grid_idx = 0
            pos = self._grid[self._grid_idx]
            self._grid_idx += 1

        self.pusher_pos = pos
        action = np.array(pos, dtype=np.float32)

        # Step the environment
        obs, reward, term, trunc, _ = e.step(action)
        self._step_count += 1
        self.last_reward = float(reward)

        if float(reward) > 0:
            self.best_reward = float(reward)
            self.phase = Phase.CONTACT
            msg = f"Contact made! Reward={reward:.3f} at ({pos[0]:.0f}, {pos[1]:.0f})"
        else:
            msg = f"Searching grid {self._grid_idx}/{len(self._grid)} at ({pos[0]:.0f}, {pos[1]:.0f})"

        frame_pil = _render_with_grid(e.render())
        uri = _frame_to_uri(frame_pil)

        return StepResult(
            image_b64=uri,
            reward=float(reward),
            pos=list(pos),
            phase=self.phase,
            message=msg,
            step=self._step_count,
        )

    def _do_contact_step(self) -> StepResult:
        """Phase 2 (first push): Gemma sees the frame and decides the first push target.

        This is a one-time transition from CONTACT -> PUSH. We ask Gemma for the
        initial push direction now that we're touching the T-block.
        """
        self.phase = Phase.PUSH
        return self._do_push_step(is_first=True)

    def _do_push_step(self, is_first: bool = False) -> StepResult:
        """Phase 2/3: Ask Gemma where to push, execute, and adapt."""
        e = self.env
        client = self._client

        # Render the frame with grid overlay
        raw_frame = e.render()
        frame_pil = _render_with_grid(raw_frame)
        uri = _frame_to_uri(frame_pil)

        # Build the prompt with feedback
        prompt = self._build_push_prompt(is_first)

        # Call Gemma
        gemma_reasoning = ""
        gemma_ms = 0.0
        target_x, target_y = 256.0, 256.0

        if client:
            t0 = time.perf_counter()
            try:
                result = client.image_chat(
                    prompt,
                    uri,
                    temperature=0.15,
                    max_tokens=200,
                    response_format={"type": "json_schema", "json_schema": PUSH_SCHEMA},
                )
                gemma_ms = (time.perf_counter() - t0) * 1000
                data = json.loads(result.content)
                target_x = float(data.get("target_x", 256.0))
                target_y = float(data.get("target_y", 256.0))
                gemma_reasoning = data.get("reasoning", "")[:100]
            except Exception as ex:
                gemma_reasoning = f"[error: {ex}]"
                # Fallback: nudge toward center
                target_x = max(0, min(512, self.pusher_pos[0] + self._guess_delta()[0]))
                target_y = max(0, min(512, self.pusher_pos[1] + self._guess_delta()[1]))
        else:
            # No client — heuristic fallback
            dx, dy = self._guess_delta()
            target_x = max(0, min(512, self.pusher_pos[0] + dx))
            target_y = max(0, min(512, self.pusher_pos[1] + dy))

        # Clamp to valid range
        target_x = max(0.0, min(SPACE_SIZE, target_x))
        target_y = max(0.0, min(SPACE_SIZE, target_y))

        # Execute the push: move pusher to target
        self.pusher_pos = [target_x, target_y]
        action = np.array(self.pusher_pos, dtype=np.float32)
        obs, reward, term, trunc, _ = e.step(action)
        self._step_count += 1

        # Compute reward delta
        reward_delta = float(reward) - self.last_reward
        self.last_action_was_good = reward_delta > 0.001
        self.last_reward = float(reward)
        if float(reward) > self.best_reward:
            self.best_reward = float(reward)

        # Record history
        self._push_history.append({
            "target": [target_x, target_y],
            "reward": float(reward),
            "delta": reward_delta,
        })

        # Build status message
        delta_str = f"+{reward_delta:.3f}" if reward_delta > 0 else f"{reward_delta:.3f}"
        if reward_delta > 0.01:
            msg = f"Good push! Reward {delta_str} (now {float(reward):.3f}) — keep going!"
        elif reward_delta > 0:
            msg = f"Slight improvement {delta_str} (now {float(reward):.3f})"
        elif reward_delta > -0.01:
            msg = f"No change ({delta_str}) — try a different direction"
        else:
            msg = f"Reward dropped {delta_str} (now {float(reward):.3f}) — change strategy"

        # Check for goal (reward >= 0.9 is very good in PushT)
        if float(reward) >= 0.8:
            self.phase = Phase.DONE
            msg = f"Goal reached! Reward={float(reward):.3f}"

        # Check for terminal
        if term or trunc:
            self.phase = Phase.DONE
            msg = f"Episode terminated. Final reward={float(reward):.3f}"

        # Render output frame
        out_pil = _render_with_grid(e.render())
        out_uri = _frame_to_uri(out_pil)

        return StepResult(
            image_b64=out_uri,
            reward=float(reward),
            pos=[target_x, target_y],
            phase=self.phase,
            message=msg,
            step=self._step_count,
            reward_delta=reward_delta,
            gemma_reasoning=gemma_reasoning,
            gemma_latency_ms=gemma_ms,
        )

    def _do_idle_step(self) -> StepResult:
        """Phase DONE: just render and report."""
        e = self.env
        frame_pil = _render_with_grid(e.render())
        uri = _frame_to_uri(frame_pil)
        return StepResult(
            image_b64=uri,
            reward=self.last_reward,
            pos=self.pusher_pos,
            phase=Phase.DONE,
            message=f"Done. Best reward: {self.best_reward:.3f}",
            step=self._step_count,
        )

    # ---- Helpers ----------------------------------------------------------

    def _build_push_prompt(self, is_first: bool) -> str:
        """Build the prompt for Gemma based on current state and history."""
        lines: list[str] = [
            "You are controlling a robotic pusher in a PushT simulation.",
            "Your task: push the T-shaped block into the green target zone.",
            "",
            "The image shows a 96x96 downsampled view with a grid overlay:",
            "  Zones: A(UL) B(UC) C(UR) D(ML) E(MC) F(MR) G(LL) H(LC) I(LR)",
            "",
            f"Pusher position: ({self.pusher_pos[0]:.0f}, {self.pusher_pos[1]:.0f})",
            f"Current contact reward: {self.last_reward:.3f}",
            f"Best reward so far: {self.best_reward:.3f}",
        ]

        if not is_first and self._push_history:
            last = self._push_history[-1]
            delta = last["delta"]
            if delta > 0.01:
                lines.append("")
                lines.append(
                    f"Last push INCREASED reward by +{delta:.3f}. "
                    "You are pushing the right way! Continue pushing the T-block "
                    "in a similar direction toward the green target."
                )
            elif delta > 0:
                lines.append("")
                lines.append(
                    f"Last push barely changed reward (+{delta:.3f}). "
                    "Try a slightly different angle or push harder."
                )
            elif delta > -0.01:
                lines.append("")
                lines.append(
                    "Last push did not change reward. The pusher may have lost contact "
                    "with the T-block. Move toward where you last saw the T-block and "
                    "re-establish contact."
                )
            else:
                lines.append("")
                lines.append(
                    f"Last push DECREASED reward by {delta:.3f}. "
                    "You pushed the wrong way. Try a different direction — move toward "
                    "where the T-block is and push it toward the green target."
                )

        lines.append("")
        lines.append(
            "You ARE touching the T-block (reward > 0 means contact). "
            "Output the XY position (0-512, 0-512) to move the pusher to "
            "in order to push the T-block toward the green target."
        )
        lines.append("Think about where the T-block is now, where the target is, "
                     "and how to push the block toward the target.")
        lines.append("Be decisive. Move to nudge the block, not to random positions.")

        return "\n".join(lines)

    def _guess_delta(self) -> tuple[float, float]:
        """Heuristic fallback when Gemma is unavailable: push toward center."""
        cx, cy = self.pusher_pos
        target_center = 256.0
        dx = max(-64.0, min(64.0, target_center - cx))
        dy = max(-64.0, min(64.0, target_center - cy))
        # If we're near center, push in a spiral pattern
        if abs(dx) < 10 and abs(dy) < 10:
            phase = (self._step_count // 3) % 4
            if phase == 0:
                dx, dy = 50.0, 0.0
            elif phase == 1:
                dx, dy = 0.0, 50.0
            elif phase == 2:
                dx, dy = -50.0, 0.0
            else:
                dx, dy = 0.0, -50.0
        return dx, dy


# ---------------------------------------------------------------------------
# Simple CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the hybrid controller on the CLI for testing / debugging."""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid PushT controller")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--video", action="store_true", help="show frame info per step")
    args = parser.parse_args()

    ctrl = HybridPushtController()
    results = ctrl.run_episode(max_steps=args.max_steps)

    print()
    print("=" * 62)
    print("  HYBRID PUSHT CONTROLLER — Episode Trace")
    print("=" * 62)
    print(f"{'Step':>5} {'Phase':>10} {'Reward':>8} {'ΔReward':>8} {'Latency':>8} {'Message'}")
    print("-" * 62)

    for r in results:
        lat = f"{r.gemma_latency_ms:.0f}ms" if r.gemma_latency_ms else " —   "
        delta = f"{r.reward_delta:+.3f}" if abs(r.reward_delta) > 0.001 else " —   "
        msg_short = r.message[:45]
        print(f"{r.step:>4}/{r.step:>4} {r.phase:>10} {r.reward:>8.3f} {delta:>8} {lat:>8} {msg_short}")

    final = results[-1] if results else None
    print()
    print(f"  Final reward:   {final.reward:.3f}" if final else "")
    print(f"  Best reward:    {ctrl.best_reward:.3f}")
    print(f"  Total steps:    {ctrl._step_count}")
    print(f"  Phase ended:    {ctrl.phase}")


if __name__ == "__main__":
    main()
