"""Tabletop simulation world for the reasoning-in-the-loop robot demo.

Pure-Python sim: objects carry ground-truth positions, a gripper moves toward
targets, and the scene renders to a PIL image (with a zone-grid overlay) that is
*exactly* what Gemma 4 perceives each tick. The model never sees coordinates —
it reads the image, names an object, and the skill layer resolves that object's
id to its precise position (the Semantic -> Geometric bridge).
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

# Tabletop dimensions in pixels (image space doubles as world space).
WIDTH = 640
HEIGHT = 420
GRID_COLS = 3
GRID_ROWS = 2
ZONE_LABELS = ["A", "B", "C", "D", "E", "F"]  # row-major over the grid


@dataclass
class SimObject:
    """A graspable object on the table. Position is ground truth, known to the
    sim — used by the skill layer, never handed to the model as text."""

    id: str
    label: str  # semantic label the model reads from the image, e.g. "red cup"
    color: tuple[int, int, int]
    x: float
    y: float
    radius: float = 26.0
    attribute: str = ""  # e.g. "cracked"

    @property
    def zone(self) -> str:
        col = min(int(self.x / (WIDTH / GRID_COLS)), GRID_COLS - 1)
        row = min(int(self.y / (HEIGHT / GRID_ROWS)), GRID_ROWS - 1)
        return ZONE_LABELS[row * GRID_COLS + col]


@dataclass
class Gripper:
    x: float
    y: float
    holding: str | None = None  # object id currently grasped
    closed: bool = False


@dataclass
class World:
    objects: dict[str, SimObject] = field(default_factory=dict)
    gripper: Gripper = field(default_factory=lambda: Gripper(WIDTH / 2, 30.0))
    bins: dict[str, tuple[float, float]] = field(default_factory=dict)
    tick: int = 0

    # -- scene construction -------------------------------------------------
    def add(self, obj: SimObject) -> None:
        self.objects[obj.id] = obj

    def add_bin(self, name: str, x: float, y: float) -> None:
        self.bins[name] = (x, y)

    def get(self, oid: str) -> SimObject | None:
        return self.objects.get(oid)

    def resolve(self, target: str) -> tuple[float, float] | None:
        """Object id or bin name -> ground-truth (x, y). The geometric half of
        the bridge: the model named the target, the sim knows where it is."""
        if target in self.objects:
            o = self.objects[target]
            return (o.x, o.y)
        if target in self.bins:
            return self.bins[target]
        return None

    def gripper_zone(self) -> str:
        g = self.gripper
        col = min(int(g.x / (WIDTH / GRID_COLS)), GRID_COLS - 1)
        row = min(int(g.y / (HEIGHT / GRID_ROWS)), GRID_ROWS - 1)
        return ZONE_LABELS[row * GRID_COLS + col]

    def zone_of(self, target: str) -> str:
        if target in self.objects:
            return self.objects[target].zone
        if target in self.bins:
            bx, by = self.bins[target]
            col = min(int(bx / (WIDTH / GRID_COLS)), GRID_COLS - 1)
            row = min(int(by / (HEIGHT / GRID_ROWS)), GRID_ROWS - 1)
            return ZONE_LABELS[row * GRID_COLS + col]
        return "none"

    # -- per-tick physics ---------------------------------------------------
    def physics(self) -> None:
        """A held object rides with the gripper."""
        if self.gripper.holding:
            held = self.objects.get(self.gripper.holding)
            if held is not None:
                held.x, held.y = self.gripper.x, self.gripper.y

    # -- rendering ----------------------------------------------------------
    def render(self) -> Image.Image:
        img = Image.new("RGB", (WIDTH, HEIGHT), (238, 238, 240))
        d = ImageDraw.Draw(img)

        cw, ch = WIDTH / GRID_COLS, HEIGHT / GRID_ROWS
        for c in range(1, GRID_COLS):
            d.line([(c * cw, 0), (c * cw, HEIGHT)], fill=(212, 212, 218), width=1)
        for r in range(1, GRID_ROWS):
            d.line([(0, r * ch), (WIDTH, r * ch)], fill=(212, 212, 218), width=1)
        for i, lab in enumerate(ZONE_LABELS):
            r, c = divmod(i, GRID_COLS)
            d.text((c * cw + 5, r * ch + 4), f"Zone {lab}", fill=(175, 175, 182))

        for name, (bx, by) in self.bins.items():
            d.rectangle([bx - 36, by - 26, bx + 36, by + 26], outline=(95, 95, 95), width=3)
            d.text((bx - 32, by - 8), name, fill=(95, 95, 95))

        for obj in self.objects.values():
            d.ellipse(
                [obj.x - obj.radius, obj.y - obj.radius, obj.x + obj.radius, obj.y + obj.radius],
                fill=obj.color,
                outline=(35, 35, 35),
                width=2,
            )
            if obj.attribute == "cracked":
                d.line(
                    [(obj.x - 11, obj.y - 13), (obj.x + 4, obj.y), (obj.x - 7, obj.y + 13)],
                    fill=(20, 20, 20),
                    width=2,
                )
            d.text((obj.x - obj.radius, obj.y + obj.radius + 3), obj.label, fill=(30, 30, 30))

        g = self.gripper
        col = (205, 45, 45) if g.closed else (45, 120, 205)
        d.line([(g.x - 17, g.y), (g.x + 17, g.y)], fill=col, width=4)
        d.line([(g.x, g.y), (g.x, g.y - 24)], fill=col, width=4)
        d.ellipse([g.x - 5, g.y - 5, g.x + 5, g.y + 5], fill=col)
        return img


def image_to_data_uri(img: Image.Image) -> str:
    """PNG-encode an in-memory PIL image to a base64 data URI for Gemma."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
