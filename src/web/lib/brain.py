"""Gemma 4 prompt construction + structured-output call.

One call = one composite image + EE proprioception + recent step history
-> structured JSON {tool, params, reasoning}.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Iterable

from src.provider import LLMProvider
from src.web.lib.imaging import img_to_b64, make_composite, overlay_grid
from src.web.lib.schema import build_intent_schema
from src.web.lib.sim import Snapshot
from src.web.lib.tasks import TaskSpec


TABLE_Z = 0.85


@dataclass
class HistoryItem:
    step: int
    tool: str
    tool_params: str
    reasoning: str
    ee_x: float
    ee_y: float
    ee_z: float
    gripper_open: bool
    verdict_note: str = ""


@dataclass
class Intent:
    tool: str
    params: dict
    reasoning: str
    latency_ms: int


def _history_text(history: Iterable[HistoryItem]) -> str:
    items = list(history)[-2:]
    if not items:
        return "(No previous steps -- this is your first action.)\n"
    lines = ["Recent steps (newest last):"]
    for h in items:
        line = (
            f"  Step {h.step}: {h.tool}"
            f"  EE=({h.ee_x:.3f},{h.ee_y:.3f},{h.ee_z:.3f}) "
        )
        if h.verdict_note:
            line += f"[{h.verdict_note}]"
        lines.append(line)
    return "\n".join(lines)


def _mode_instructions(spec: TaskSpec | None) -> str:
    if spec is None:
        return ""
    m = spec.mode
    if m == "door":
        return (
            "\n\nDOOR task: Grip the handle firmly (close gripper), "
            "then pull the door open by moving the arm backward (negative y). "
            "The handle position is listed above. Pull until the hinge angle exceeds 0.5 rad.\n"
        )
    if m == "wipe":
        return (
            "\n\nWIPE task: No gripper needed. Move the end effector across the table "
            "surface at z ~ 0.90 in overlapping passes. Cover as much area as possible.\n"
        )
    if m == "nut_assembly":
        return (
            "\n\nNUT ASSEMBLY: Pick up the square nut from the table and place it onto "
            "the vertical peg. Descend low to grasp (nut z ~ 0.89), close gripper, lift, "
            "align over the peg center, then lower gently.\n"
        )
    if m == "clear_table":
        return (
            "\n\nCLEAR TABLE: Pick up each object ONE AT A TIME and place it in the right bin at "
            "(0.10, 0.28). Order: Can -> Milk -> Bread -> Cereal.\n"
            "Check progress below -- objects marked PLACED \u2713 are done. "
            "Focus on the first unplaced object.\n"
        )
    if m == "pick_place":
        return (
            "\n\nPick and place: after lift, move xy to drop zone at (0.10, 0.28), "
            "then descend and OPEN gripper to release.\n"
        )
    if m == "stack":
        return (
            "\n\nSTACK: After grasping the red cube, lift it, then move it directly "
            "above the green cube (its position is listed). Descend until the red cube "
            "rests on the green cube, then open the gripper.\n"
        )
    return ""


def _objects_text(snap: Snapshot, spec: TaskSpec, prev_snap: Snapshot | None = None) -> str:
    if not snap.objects:
        return "(No object positions yet -- rely on the image.)\n"
    
    bin_xy = spec.place_xy if spec else None
    placed_keys: set[str] = set()
    if bin_xy and spec and spec.compound_objects:
        bx, by = bin_xy
        for obj_key in spec.compound_objects:
            if obj_key in snap.objects:
                opos = snap.objects[obj_key]
                dist_to_bin = ((float(opos[0]) - bx)**2 + (float(opos[1]) - by)**2)**0.5
                if dist_to_bin < 0.10 and float(opos[2]) < 0.91:
                    placed_keys.add(obj_key)

    lines = ["OBJECTS (physics ground truth):"]
    label_for = dict(spec.visible_objects) if spec else {}
    target_obj = spec.target_object if spec else None
    
    for key, pos in snap.objects.items():
        label = label_for.get(key, key)
        is_placed = key in placed_keys
        
        if is_placed:
            marker = " PLACED \u2713"
        elif key == target_obj:
            marker = " <<< TARGET <<<"
        else:
            marker = ""
        
        changed = ""
        if prev_snap is not None and key in prev_snap.objects:
            old = prev_snap.objects[key]
            dx = float(pos[0]) - float(old[0])
            dy = float(pos[1]) - float(old[1])
            dist = (dx**2 + dy**2)**0.5
            if dist > 0.01:
                changed = f" \u25c4 CHANGED ({dist*100:.1f}cm)"
        # Pre-computed distance from EE to this object
        ox, oy, oz = float(pos[0]), float(pos[1]), float(pos[2])
        ex, ey, ez = float(snap.ee_pos[0]), float(snap.ee_pos[1]), float(snap.ee_pos[2])
        dxy = ((ox - ex)**2 + (oy - ey)**2)**0.5
        dz_cm = (oz - ez) * 100
        if dz_cm < 0:
            dir_hint = f"below ({abs(dz_cm):.1f}cm below EE)"
        else:
            dir_hint = f"descend ({dz_cm:.1f}cm above EE)"
        dist_str = f" | XY {dxy*100:.1f}cm, Z {dir_hint}"
        lines.append(f"  {label}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}){marker}{dist_str}{changed}")
    
    # Clear table progress
    if spec and spec.mode == "clear_table" and spec.compound_objects:
        remaining = [label_for.get(k, k) for k in spec.compound_objects if k not in placed_keys]
        if remaining:
            lines.append(f"  \u2192 Still to place: {', '.join(remaining)}")
        else:
            lines.append(f"  \u2192 All objects placed in bin!")
    
    # Holding status: check if gripper has object
    if not snap.gripper_open:
        for key, pos in snap.objects.items():
            if float(pos[2]) > TABLE_Z + 0.02:
                label = label_for.get(key, key)
                lines.append(f"HOLDING: {label} (confirmed, z={pos[2]:.3f})")
                break
        else:
            lines.append("HOLDING: Nothing (gripper CLOSED but empty)")
    else:
        lines.append("HOLDING: Nothing (gripper OPEN)")
    
    if not snap.gripper_open:
        for key, pos in snap.objects.items():
            if float(pos[2]) > 0.87:
                label = dict(spec.visible_objects).get(key, key)
                lines.append(f"HOLDING: {label}")
                break
        else:
            lines.append("HOLDING: Nothing (gripper closed but empty)")
    else:
        lines.append("HOLDING: Nothing (gripper open)")
    return "\n".join(lines) + "\n"


def _prompt(task: str, snap: Snapshot, history: Iterable[HistoryItem], spec: TaskSpec | None = None, prev_snap: Snapshot | None = None, obj_block_override: str | None = None) -> str:
    obj_block = obj_block_override if obj_block_override is not None else (_objects_text(snap, spec, prev_snap) if spec is not None else "")
    return (
        "You control a Franka Panda 7-DOF robot arm.\n\n"
        "IMAGE: Two rows. Top: [birdview | frontview] full res for positioning. "
        "Bottom center: eye_in_hand gripper close-up. Zone grid overlaid on birdview.\n\n"
        "YOUR ARM (proprioception):\n"
        f"  End-effector: ({snap.ee_pos[0]:.3f}, {snap.ee_pos[1]:.3f}, {snap.ee_pos[2]:.3f})\n"
        f"  Gripper: {'OPEN' if snap.gripper_open else 'CLOSED'}\n\n"
        f"{obj_block}"
        f"TASK: {task}\n\n"
        f"{_history_text(history)}\n"

        "WORKFLOW (descend below top before closing, then lift slowly):\n"
        "  1.reach OPEN  at (ox, oy, oz+0.10)\n"
        "  2.descend OPEN at (ox, oy, oz+0.02)\n"
        "  3.grasp CLOSE at (ox, oy, oz-0.01) x2\n"
        "  4.lift CLOSE  to oz+0.10 (+0.20 if stuck)\n"
        "  5.done only after verifier confirms.\n\n"
        f"{_mode_instructions(spec)}"

        "OUTPUT: tool, params, reasoning.\n"
        "HINT: For move_to, use target='ObjectName' instead of x/y/z -- coordinates auto-resolve.\n"
    )


class GemmaBrain:

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._client = provider

    def _ensure_provider(self) -> LLMProvider:
        if self._client is None:
            from src.provider import ProviderRegistry
            self._client = ProviderRegistry.default()
        return self._client

    def think(self, task: str, snap: Snapshot, history: Iterable[HistoryItem], spec: TaskSpec | None = None, prev_snap: Snapshot | None = None, send_image: bool = True, vision_text_override: str | None = None) -> Intent:
        t0 = time.perf_counter()

        # When vision_text_override is set, use it INSTEAD of _objects_text().
        # This is the blinding layer: Gemma sees camera-derived positions, not
        # ground-truth simulator object positions.
        obj_block = vision_text_override if vision_text_override is not None else (_objects_text(snap, spec, prev_snap) if spec is not None else "")

        if send_image:
            composite_b64 = img_to_b64(
                make_composite(
                    overlay_grid(snap.birdview),
                    snap.frontview,
                    snap.eye_in_hand,
                )
            )
            result = self._ensure_provider().image_chat(
                prompt=_prompt(task, snap, history, spec, prev_snap, obj_block_override=vision_text_override),
                image_b64=composite_b64,
                temperature=0.0,
                seed=42,
                max_tokens=300,
                response_format={"type": "json_schema", "json_schema": build_intent_schema()},
            )
        else:
            # Text-only call — no image, saves ~300ms
            result = self._ensure_provider().chat(
                messages=[{"role": "user", "content": _prompt(task, snap, history, spec, prev_snap, obj_block_override=vision_text_override)}],
                temperature=0.0,
                seed=42,
                max_tokens=300,
                response_format={"type": "json_schema", "json_schema": build_intent_schema()},
            )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        raw = result.content.strip()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            # Handle truncated JSON: try to close unterminated strings and braces
            import re
            fixed = raw
            # If reasoning string is unterminated, close it
            if fixed.count('"') % 2 == 1:
                fixed += '"'
            # Add closing braces
            open_braces = fixed.count("{") - fixed.count("}")
            if open_braces > 0:
                fixed += "}" * open_braces
            # Add closing brackets
            open_brackets = fixed.count("[") - fixed.count("]")
            if open_brackets > 0:
                fixed += "]" * open_brackets
            try:
                d = json.loads(fixed)
            except json.JSONDecodeError:
                # Last resort: extract tool name with regex
                m = re.search(r'"tool"\s*:\s*"(\w+)"', raw)
                tool = m.group(1) if m else "move_to"
                d = {"tool": tool, "params": {}, "reasoning": "parse_fallback"}
        tool = d.get("tool", "move_to")
        params = d.get("params", {})
        return Intent(
            tool=tool,
            params=params,
            reasoning=d.get("reasoning", ""),
            latency_ms=latency_ms,
        )
