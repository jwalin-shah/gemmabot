"""Robot controller — simulates robot hardware actions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionResult:
    action: str
    target: str
    status: str  # "executed" | "skipped" | "failed"
    message: str = ""
    duration_ms: float = 0.0


class RobotController:
    """Simulates a robot executing action commands.

    In a real deployment this would call ROS2 / hardware SDK.
    For the demo it prints actions with timing."""

    def __init__(self, name: str = "GemmaBot-1") -> None:
        self._name = name
        self._position: tuple[float, float] = (0.0, 0.0)
        self._gripper_open = True

    def execute(self, action: str, target: str, **params: str | float) -> ActionResult:
        """Execute a single robot action."""
        import time

        start = time.perf_counter()

        if action == "move_to":
            self._position = (self._position[0] + 0.5, self._position[1])
            status, msg = "executed", f"Moved toward {target}"
        elif action == "pick_up":
            if not self._gripper_open:
                status, msg = "failed", "Gripper already holding something"
            else:
                self._gripper_open = False
                status, msg = "executed", f"Picked up {target}"
        elif action == "place":
            self._gripper_open = True
            status, msg = "executed", f"Placed {target}"
        elif action == "push":
            status, msg = "executed", f"Pushed {target}"
        elif action == "pause":
            status, msg = "executed", "Pausing"
        elif action == "navigate_home":
            self._position = (0.0, 0.0)
            status, msg = "executed", "Returned to home"
        else:
            status, msg = "skipped", f"Unknown action: {action}"

        elapsed = (time.perf_counter() - start) * 1000
        return ActionResult(action=action, target=target, status=status, message=msg, duration_ms=elapsed)

    @property
    def name(self) -> str:
        return self._name

    @property
    def position(self) -> tuple[float, float]:
        return self._position