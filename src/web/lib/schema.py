"""Single source of truth for tool schemas and validation.

- TOOLS: canonical enum of all tools the LLM can invoke.
- TOOL_PARAMS: per-tool parameter definitions with types and required flags.
- build_intent_schema(): generates the JSON schema dict for LLM structured output.
- validate_tool() / validate_tool_params(): runtime validation helpers.
"""

from __future__ import annotations

# ── Canonical tool list ──────────────────────────────────────────────────

TOOLS: list[str] = ["move_to", "grasp", "grasp_side", "lift", "place", "done"]

# ── Per-tool parameter definitions ───────────────────────────────────────
# Each entry maps parameter name -> {type, description, required, enum?}.
TOOL_PARAMS: dict[str, dict[str, dict]] = {
    "move_to": {
        "target": {"type": "string", "description": "Object name to move toward (e.g. 'Can', 'Milk', 'cube'). Use INSTEAD of x/y/z.", "required": False},
        "x": {"type": "number", "description": "X coordinate for move_to/place", "required": False},
        "y": {"type": "number", "description": "Y coordinate for move_to/place", "required": False},
        "z": {"type": "number", "description": "Z height for move_to", "required": False},
    },
    "grasp": {
        "object_name": {"type": "string", "description": "Object name for grasp: Can, Milk, Bread, Cereal, cube, cubeA, SquareNut", "required": True},
    },
    "grasp_side": {
        "object_name": {"type": "string", "description": "Object name to grasp from the side", "required": True},
        "direction": {"type": "string", "enum": ["above", "left", "right", "front", "back"], "description": "Approach direction for grasp_side", "required": True},
    },
    "lift": {
        "height": {"type": "number", "description": "Lift height in meters (default 0.15)", "required": False},
    },
    "place": {
        "x": {"type": "number", "description": "X coordinate for placement", "required": True},
        "y": {"type": "number", "description": "Y coordinate for placement", "required": True},
        "z": {"type": "number", "description": "Z height override for placement (default table surface)", "required": False},
    },
    "done": {},
}


# ── LLM JSON schema generation ──────────────────────────────────────────

def build_intent_schema() -> dict:
    """Build the JSON schema dict for LLM structured-output tool calling.

    Returns the same schema that was previously inline in brain.py as
    ``INTENT_SCHEMA``.  All parameter definitions from ``TOOL_PARAMS`` are
    unioned into a single flat ``params`` object, since the LLM outputs
    one schema regardless of which tool it chooses.
    """
    params_properties: dict[str, dict] = {}
    for tool_params in TOOL_PARAMS.values():
        for pname, pdef in tool_params.items():
            if pname in params_properties:
                continue
            prop: dict = {"description": pdef["description"]}
            prop["type"] = pdef["type"]
            if "enum" in pdef:
                prop["enum"] = pdef["enum"]
            params_properties[pname] = prop

    return {
        "name": "tool_call",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tool": {
                    "type": "string",
                    "enum": list(TOOLS),
                    "description": "Which tool to use. done = task complete.",
                },
                "params": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": params_properties,
                },
                "reasoning": {"type": "string"},
            },
            "required": ["tool", "params", "reasoning"],
        },
    }


# ── Runtime validation ──────────────────────────────────────────────────

def validate_tool(tool_name: str) -> bool:
    """Return True if *tool_name* is a valid tool name."""
    return tool_name in TOOLS


def validate_tool_params(tool_name: str, params: dict) -> list[str]:
    """Validate *params* for the given *tool_name*.

    Returns a list of error strings (empty list = valid).
    """
    errors: list[str] = []
    if tool_name not in TOOL_PARAMS:
        errors.append(f"Unknown tool: '{tool_name}'")
        return errors
    param_defs = TOOL_PARAMS[tool_name]

    # Check required params are present.
    for pname, pdef in param_defs.items():
        if pdef.get("required", False) and pname not in params:
            errors.append(f"Missing required parameter '{pname}' for tool '{tool_name}'")

    # Check supplied params are valid.
    for pname, pval in params.items():
        if pname not in param_defs:
            errors.append(f"Unexpected parameter '{pname}' for tool '{tool_name}'")
            continue
        ptype = param_defs[pname]["type"]
        if ptype == "number" and not isinstance(pval, (int, float)):
            errors.append(f"Parameter '{pname}' for tool '{tool_name}' must be a number, got {type(pval).__name__}")
        elif ptype == "string" and not isinstance(pval, str):
            errors.append(f"Parameter '{pname}' for tool '{tool_name}' must be a string, got {type(pval).__name__}")
        if "enum" in param_defs[pname] and pval not in param_defs[pname]["enum"]:
            errors.append(f"Parameter '{pname}' for tool '{tool_name}' must be one of {param_defs[pname]['enum']}, got '{pval}'")

    return errors
