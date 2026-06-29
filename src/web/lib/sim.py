"""Robosuite environment lifecycle + observation snapshots.

Keeps the FastAPI layer free of robosuite imports until first use, so the
process can start (and serve the landing page) even if MuJoCo isn't warm yet.

Camera arrays on a Snapshot are pre-oriented (flipped to PIL/HTML
convention) -- consumers do not need to call ``fix_img`` again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.web.lib.imaging import fix_img
from src.web.lib.tasks import DEFAULT_TASK, TaskSpec, get as _get_task, object_positions
from src.config import CAMERA_HEIGHT, CAMERA_WIDTH


_log = logging.getLogger(__name__)

GRIPPER_OPEN_THRESHOLD = 0.030


@dataclass
class Snapshot:
    """Single-frame view of the robot -- arrays are already PIL-oriented."""
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    gripper_qpos: float
    gripper_open: bool
    birdview: np.ndarray
    frontview: np.ndarray
    eye_in_hand: np.ndarray
    objects: dict[str, np.ndarray] = field(default_factory=dict)
    object_quats: dict[str, np.ndarray] = field(default_factory=dict)
    state: dict[str, float] = field(default_factory=dict)


class PandaSim:
    """Wrap a single robosuite env so callers do not touch globals."""

    def __init__(self, task: TaskSpec | str | None = None) -> None:
        self._env: Any | None = None
        self._task: TaskSpec = _resolve(task or DEFAULT_TASK)

    @property
    def task(self) -> TaskSpec:
        return self._task

    def env(self) -> Any:
        if self._env is None:
            import robosuite as suite

            self._env = suite.make(
                self._task.env_name,
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=["birdview", "frontview", "robot0_eye_in_hand"],
                camera_heights=CAMERA_HEIGHT,
                camera_widths=CAMERA_WIDTH,
            )
            self._env.reset()
        return self._env

    def set_task(self, task: TaskSpec | str) -> None:
        new_spec = _resolve(task)
        if new_spec.key == self._task.key and self._env is not None:
            return
        self._close()
        self._task = new_spec

    def reset(self) -> Snapshot:
        self._close()
        return self.snapshot()

    def snapshot(self) -> Snapshot:
        obs = self.env()._get_observations(force_update=True)
        return _make_snapshot(obs, self._task)

    def step(self, action: np.ndarray) -> Snapshot:
        obs, _reward, done, _ = self.env().step(action)
        if done:
            self.env().reset()
        return _make_snapshot(obs, self._task)

    def _close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                _log.exception("env.close failed")
            self._env = None


def _make_snapshot(obs: dict[str, Any], spec: TaskSpec) -> Snapshot:
    ee = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)))
    ee_quat = np.asarray(obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0])))
    grip_qpos_arr = obs.get("robot0_gripper_qpos", np.array([0.0]))
    grip_qpos = float(grip_qpos_arr[0]) if len(grip_qpos_arr) > 0 else 0.0
    env_state: dict[str, float] = {}
    if spec.mode == "door":
        env_state["hinge_qpos"] = float(obs.get("hinge_qpos", 0.0))
    if spec.mode == "wipe":
        env_state["proportion_wiped"] = float(obs.get("proportion_wiped", 0.0))
    return Snapshot(
        ee_pos=ee,
        ee_quat=ee_quat,
        gripper_qpos=grip_qpos,
        gripper_open=grip_qpos > GRIPPER_OPEN_THRESHOLD,
        birdview=fix_img(obs.get("birdview_image")),
        frontview=fix_img(obs.get("frontview_image")),
        eye_in_hand=fix_img(obs.get("robot0_eye_in_hand_image")),
        objects=object_positions(obs, spec),
        object_quats=object_quaternions(obs, spec),
        state=env_state,
    )


def _resolve(task: TaskSpec | str) -> TaskSpec:
    return task if isinstance(task, TaskSpec) else _get_task(task)


def object_quaternions(obs: dict, spec: TaskSpec) -> dict[str, np.ndarray]:
    """Extract WXYZ quaternions for visible objects."""
    out: dict[str, np.ndarray] = {}
    for key, _label in spec.visible_objects:
        arr = obs.get(f"{key}_quat")
        if arr is not None:
            out[key] = np.asarray(arr, dtype=float)
    return out
