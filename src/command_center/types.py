"""Type definitions for the Command Center multi-agent architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Types of signals the Command Center root can receive."""
    IMAGE = "image"
    TEXT = "text"       # voice transcribed to text
    VOICE = "voice"     # raw audio (future)
    SENSOR = "sensor"   # temperature, distance, etc.
    TIMER = "timer"     # periodic tick
    SYSTEM = "system"   # internal events


class Branch(Enum):
    """All registered branches in the Command Center."""
    VISION = "vision"
    ACTION_PLANNER = "action_planner"
    SAFETY = "safety"
    COORDINATOR = "coordinator"
    SUMMARIZER = "summarizer"
    SPEED_BENCH = "speed_bench"
    ORACLE = "oracle"  # general reasoning / ask


class Urgency(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CommandCenterSignal:
    """A signal entering the Command Center — could be an image, text command, sensor read, etc."""
    type: SignalType
    payload: Any
    source: str = "user"
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import time
        if not self.timestamp:
            self.timestamp = time.perf_counter()


@dataclass
class CommandCenterObservation:
    """What the Command Center root sees after processing a signal."""
    signal: CommandCenterSignal
    summary: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    hazards: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class RoutingDecision:
    """The root's decision about what to do with an observation."""
    route_to: list[Branch] = field(default_factory=list)
    priority: Urgency = Urgency.LOW
    instruction: str = ""
    parallel: bool = True  # dispatch to all branches in parallel?
    context_hint: str = ""
    command: dict[str, str] = field(default_factory=dict)  # action, target, reasoning


@dataclass
class BranchOutput:
    """Output from a single branch agent."""
    branch: Branch
    content: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class CommandCenterCommand:
    """A command synthesized by the Command Center for execution."""
    action: str  # e.g. "move_to", "pick_up", "speak", "wait"
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class CommandCenterLoopResult:
    """Complete result of one Treehouse loop iteration."""
    signal: CommandCenterSignal
    observation: CommandCenterObservation | None = None
    decision: RoutingDecision | None = None
    branch_outputs: list[BranchOutput] = field(default_factory=list)
    commands: list[CommandCenterCommand] = field(default_factory=list)
    total_latency_ms: float = 0.0
    router_latency_ms: float = 0.0  # time Gemma 4 spent routing
    iterations: int = 1
