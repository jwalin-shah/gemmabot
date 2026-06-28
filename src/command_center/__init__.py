"""Command Center — tree-based multi-agent orchestration powered by Cerebras speed.

The Command Center is a hierarchical agent system where **Gemma 4 31B sits at the root**
as an always-on watcher/router. It continuously watches incoming signals (images, text,
voice), delegates to specialist branch agents, collects results, and
decides next actions — all in real-time.

Key insight: On GPU (1-5s/call) a router adds unbearable latency. On Cerebras
(50-150ms/call) the router runs synchronously in the hot path, enabling true
real-time multi-agent coordination at 5-10 Hz.
"""

from __future__ import annotations

from src.command_center.root import CommandCenterRoot
from src.command_center.branches import BranchRegistry
from src.command_center.types import CommandCenterCommand, CommandCenterObservation, RoutingDecision

__all__ = [
    "CommandCenterRoot",
    "BranchRegistry",
    "CommandCenterCommand",
    "CommandCenterObservation",
    "RoutingDecision",
]
