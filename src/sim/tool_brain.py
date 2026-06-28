"""Tool-calling brain using Gemma 4 native function calling.

Instead of asking the model to output a JSON string (which must be parsed), this
brain defines robot actions as OpenAI-compatible tool definitions and lets Gemma 4
call them directly via the Cerebras tools API.
"""

from __future__ import annotations

import json
import time

from PIL import Image

from src.client import CerebrasClient
from src.sim.brain import Decision
from src.sim.world import image_to_data_uri


TOOLS = [
