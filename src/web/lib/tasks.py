"""Task registry — what robosuite envs are available + how to verify them.

A TaskSpec carries everything the rest of the stack needs to switch the world
and judge what happened:
  - which robosuite env to make()
  - which observation keys identify the manipulable objects
  - the success predicate, expressed in physics ground truth, NOT Gemma's
    self-reported "stage"

We deliberately keep this declarative — the verifier reads the spec and runs
plain numpy on the obs dict; no per-task code paths elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


REACH_TOL = 0.05
GRASP_TOL = 0.04
LIFT_HEIGHT = 0.03
PLACE_TOL = 0.08
DOOR_OPEN_ANGLE = 0.5
WIPE_THRESHOLD = 0.7
NUT_ON_PEG_Z = 0.92
BIN_PLACE_XY = (0.10, 0.28)
TABLE_Z = 0.85


@dataclass
class TaskSpec:
    key: str
    label: str
    description: str
    env_name: str
    target_object: str | None = None
    object_label: str = "the target object"
    place_xy: tuple[float, float] | None = None
    visible_objects: list[tuple[str, str]] = field(default_factory=list)
    secondary_object: str | None = None
    mode: Literal["pick_place", "lift", "stack", "door", "wipe", "nut_assembly", "clear_table"] = "pick_place"
    compound_objects: list[str] | None = None


_PICK_PLACE_OBJECTS = [
    ("Milk", "the milk carton"),
    ("Bread", "the loaf of bread"),
    ("Cereal", "the cereal box"),
    ("Can", "the soda can"),
]


def _pick_place_task(target: str, label_obj: str) -> TaskSpec:
    return TaskSpec(
        key=f"pick_{target.lower()}",
        label=f"Pick and place {label_obj}",
        description=f"Pick up {label_obj} from the left bin and place it in the right bin.",
        env_name="PickPlace",
        target_object=target,
        object_label=label_obj,
        place_xy=BIN_PLACE_XY,
        visible_objects=_PICK_PLACE_OBJECTS,
        mode="pick_place",
    )


TASKS: dict[str, TaskSpec] = {
    "pick_can":    _pick_place_task("Can",    "the soda can"),
    "pick_milk":   _pick_place_task("Milk",   "the milk carton"),
    "pick_bread":  _pick_place_task("Bread",  "the loaf of bread"),
    "pick_cereal": _pick_place_task("Cereal", "the cereal box"),
    "lift_cube": TaskSpec(
        key="lift_cube",
        label="Lift the red cube",
        description="Grasp the small red cube and lift it at least 5 cm above the table.",
        env_name="Lift",
        target_object="cube",
        object_label="the red cube",
        place_xy=None,
        visible_objects=[("cube", "the red cube")],
        mode="lift",
    ),
    "stack_cubes": TaskSpec(
        key="stack_cubes",
        label="Stack red on green",
        description="Pick up the red cube and stack it on top of the green cube.",
        env_name="Stack",
        target_object="cubeA",
        object_label="the red cube",
        secondary_object="cubeB",
        visible_objects=[("cubeA", "the red cube"), ("cubeB", "the green cube")],
        mode="stack",
    ),
    "open_door": TaskSpec(
        key="open_door",
        label="Open the cabinet door",
        description="Grip the door handle and pull the cabinet door open wide enough to reach inside.",
        env_name="Door",
        target_object="door",
        object_label="the cabinet door",
        place_xy=None,
        visible_objects=[("door", "the cabinet door"), ("handle", "the door handle")],
        mode="door",
    ),
    "wipe_table": TaskSpec(
        key="wipe_table",
        label="Wipe the table clean",
        description="Wipe the entire table surface clean by moving the end effector across the tabletop in overlapping passes.",
        env_name="Wipe",
        target_object=None,
        object_label="the table surface",
        place_xy=None,
        visible_objects=[],
        mode="wipe",
    ),
    "nut_assembly": TaskSpec(
        key="nut_assembly",
        label="Put the square nut on the peg",
        description="Pick up the square nut from the table and place it onto the vertical peg - like putting a dish on a drying rack.",
        env_name="NutAssemblySquare",
        target_object="SquareNut",
        object_label="the square nut",
        place_xy=None,
        visible_objects=[("SquareNut", "the square nut")],
        mode="nut_assembly",
    ),
    "clear_table": TaskSpec(
        key="clear_table",
        label="Clear all items from the table",
        description="Clear the table: pick up ALL four items one by one and place each into the right bin.",
        env_name="PickPlace",
        target_object="Can",
        object_label="all items (can, milk, bread, cereal)",
        place_xy=BIN_PLACE_XY,
        visible_objects=_PICK_PLACE_OBJECTS,
        mode="clear_table",
        compound_objects=["Can", "Milk", "Bread", "Cereal"],
    ),
}

DEFAULT_TASK = "pick_can"


def get(key: str) -> TaskSpec:
    if key not in TASKS:
        raise KeyError(f"unknown task {key!r}; known: {sorted(TASKS)}")
    return TASKS[key]


def list_specs() -> list[dict]:
    return [
        {"key": t.key, "label": t.label, "description": t.description, "mode": t.mode}
        for t in TASKS.values()
    ]


def object_positions(obs: dict, spec: TaskSpec) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, _label in spec.visible_objects:
        arr = obs.get(f"{key}_pos")
        if arr is not None:
            out[key] = np.asarray(arr, dtype=float)
    return out
