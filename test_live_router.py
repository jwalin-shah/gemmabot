#!/usr/bin/env python3
"""Live test of the Command Center Router against the Cerebras Gemma 4 API.

Tests:
  1. Text-only signal routing
  2. Image signal routing (multimodal + structured outputs)
  3. Parallel branch dispatch

Writes findings to docs/research/13-router-live-test.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from src.client import CerebrasClient
from src.command_center.root import CommandCenterRoot
from src.command_center.branches import BranchRegistry
from src.command_center.types import (
    Branch,
    BranchOutput,
    CommandCenterSignal,
    SignalType,
    Urgency,
)
from src import encode_image

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR / "docs" / "research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESEARCH_DIR / "13-router-live-test.md"

IMAGE_PATH = SCRIPT_DIR / "examples" / "images" / "workspace.jpg"


def log(msg: str) -> None:
    print(f"[TEST] {msg}")


def hr(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ======================================================================
#  PREP: Instantiate the actual Cerebras client and branch registry
# ======================================================================
hr("PREP")

log("Creating CerebrasClient...")
t0 = time.perf_counter()
client = CerebrasClient()
log(f"Client created in {(time.perf_counter() - t0)*1000:.1f}ms")

log("Creating BranchRegistry...")
registry = BranchRegistry(client)

log("Creating CommandCenterRoot...")
root = CommandCenterRoot(client, registry)
log(f"Branches registered: {root.stats['branches_registered']}")
log(f"Root ready. Avg router ms: {root.stats['avg_router_ms']}")

# ======================================================================
#  TEST 1: Text-only signal
# ======================================================================
hr("TEST 1: Text-only signal routing")

signal_text = CommandCenterSignal(
    type=SignalType.TEXT,
    payload="scan the room for hazards",
    source="user_test",
)

t1 = time.perf_counter()
result_text = root.watch(signal_text)
latency_text = (time.perf_counter() - t1) * 1000

log(f"Total latency: {latency_text:.1f}ms")
log(f"Router latency: {result_text.router_latency_ms:.1f}ms")

if result_text.decision:
    log(f"Decision: route_to={[b.value for b in result_text.decision.route_to]}, "
         f"parallel={result_text.decision.parallel}, "
         f"priority={result_text.decision.priority.value}")
    log(f"Instruction: {result_text.decision.instruction[:120]}...")
    log(f"Observation: {result_text.observation.summary if result_text.observation else 'NONE'}")
else:
    log("WARNING: No routing decision returned!")

if result_text.commands:
    for cmd in result_text.commands:
        log(f"Command: action={cmd.action}, target={cmd.target}, reason={cmd.reasoning[:60]}...")
else:
    log("WARNING: No commands returned!")

if result_text.branch_outputs:
    for bo in result_text.branch_outputs:
        log(f"  Branch {bo.branch.value}: {len(bo.content)} chars, {bo.latency_ms:.1f}ms, "
             f"err={bo.error}")
else:
    log("WARNING: No branch outputs returned!")

test1_passed = result_text.decision is not None and len(result_text.commands) > 0
log(f"TEST 1 {'PASSED' if test1_passed else 'FAILED'}")

# Reset root for clean test
client2 = CerebrasClient()
registry2 = BranchRegistry(client2)
root2 = CommandCenterRoot(client2, registry2)

# ======================================================================
#  TEST 2: Image signal
# ======================================================================
hr("TEST 2: Image signal routing (multimodal + structured outputs)")

log(f"Encoding image: {IMAGE_PATH}")
t_img = time.perf_counter()
image_b64 = encode_image(str(IMAGE_PATH))
log(f"Image encoded in {(time.perf_counter() - t_img)*1000:.1f}ms")
log(f"Base64 length: {len(image_b64)} chars")

signal_image = CommandCenterSignal(
    type=SignalType.IMAGE,
    payload=image_b64,
    source="camera",
    metadata={"task": "scan the room for hazards"},
)

t2 = time.perf_counter()
result_image = root2.watch(signal_image)
latency_image = (time.perf_counter() - t2) * 1000

log(f"Total latency: {latency_image:.1f}ms")
log(f"Router latency: {result_image.router_latency_ms:.1f}ms")

if result_image.decision:
    log(f"Decision: route_to={[b.value for b in result_image.decision.route_to]}, "
         f"parallel={result_image.decision.parallel}, "
         f"priority={result_image.decision.priority.value}")
    log(f"Instruction: {result_image.decision.instruction[:150]}...")
    log(f"Observation: {result_image.observation.summary[:150] if result_image.observation else 'NONE'}...")
else:
    log("FAIL: No routing decision returned for image signal!")

if result_image.commands:
    for cmd in result_image.commands:
        log(f"Command: action={cmd.action}, target={cmd.target}, reason={cmd.reasoning[:80]}...")
else:
    log("FAIL: No commands returned for image signal!")

if result_image.branch_outputs:
    for bo in result_image.branch_outputs:
        log(f"  Branch {bo.branch.value}: {len(bo.content)} chars, {bo.latency_ms:.1f}ms, "
             f"err={bo.error}")
else:
    log("INFO: No branch outputs (branches may not have been dispatched)")

test2_passed = result_image.decision is not None
log(f"TEST 2 {'PASSED' if test2_passed else 'FAILED'}")

# ======================================================================
#  TEST 3: Parallel branch dispatch
# ======================================================================
hr("TEST 3: Parallel branch dispatch (3 branches)")

branches_to_run = [Branch.VISION, Branch.SUMMARIZER, Branch.ORACLE]
prompts = {
    Branch.VISION: "Describe this scene for a home robot.",
    Branch.SUMMARIZER: "Summarize what a robot should know about navigating an office.",
    Branch.ORACLE: "What are the top 3 safety rules for a home robot?",
}
context = {
    "image_b64": image_b64,
    "signal_type": "image",
    "signal_source": "camera_test",
}

log(f"Dispatching {len(branches_to_run)} branches in parallel...")
t3 = time.perf_counter()
parallel_outputs = registry2.run_parallel(branches_to_run, prompts, context=context)
latency_parallel = (time.perf_counter() - t3) * 1000

log(f"Total parallel latency: {latency_parallel:.1f}ms")
for bo in parallel_outputs:
    log(f"  Branch {bo.branch.value}: {len(bo.content)} chars in {bo.latency_ms:.1f}ms, "
         f"err={bo.error}")
    if bo.content:
        log(f"    Preview: {bo.content[:120]}...")

# Check overlap: if sequential, sum of latencies >> wall time
sequential_estimate = sum(bo.latency_ms for bo in parallel_outputs)
overlap_ratio = sequential_estimate / latency_parallel if latency_parallel > 0 else 0
log(f"Sequential estimate: {sequential_estimate:.1f}ms")
log(f"Overlap ratio (sequential/parallel): {overlap_ratio:.2f}x")
test3_passed = overlap_ratio > 1.3  # Should be at least 1.3x if parallel
log(f"TEST 3 {'PASSED' if test3_passed else 'NOTE: overlapped'} (ratio={overlap_ratio:.2f}x)")

# ======================================================================
#  RAW API TESTS (fallback if structured outputs fail)
# ======================================================================
hr("TEST 4: Raw API - Can Gemma 4 return valid JSON for scene description?")

log("Sending raw text-only request for scene description (NO structured outputs)...")
raw_messages = [
    {"role": "system", "content": "Return a JSON object with fields: objects (list), layout (string), hazards (list). Be concise."},
    {"role": "user", "content": "Describe a typical home office with a desk, chair, laptop, and coffee mug."},
]

t4 = time.perf_counter()
raw_result = client.chat(raw_messages, temperature=0.1, max_tokens=512)
latency_raw = (time.perf_counter() - t4) * 1000

log(f"Raw latency: {latency_raw:.1f}ms")
log(f"Raw response ({len(raw_result.content)} chars):")
log(f"  {raw_result.content[:500]}")

# Try parsing JSON
raw_json = None
try:
    raw_json = json.loads(raw_result.content.strip().removeprefix("```json").removesuffix("```").strip())
    log("RAW JSON: Valid JSON parsed successfully!")
    log(f"  Fields: {list(raw_json.keys())}")
except (json.JSONDecodeError, Exception) as e:
    log(f"RAW JSON: Parse failed: {e}")
    # Try to find JSON in the response
    import re
    json_match = re.search(r'\{.*\}', raw_result.content, re.DOTALL)
    if json_match:
        try:
            raw_json = json.loads(json_match.group())
            log(f"  Extracted JSON via regex: fields={list(raw_json.keys())}")
        except json.JSONDecodeError:
            log("  Could not parse even after regex extraction")

test4_passed = raw_json is not None
log(f"TEST 4 {'PASSED' if test4_passed else 'FAILED'}")

# ======================================================================
#  TEST 5: Structured outputs (json_schema) via raw API
# ======================================================================
hr("TEST 5: Raw API - Structured outputs (json_schema response_format)")

log("Testing json_schema response_format...")

test_schema = {
    "name": "scene_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Objects visible in the scene",
            },
            "layout": {"type": "string", "description": "Spatial layout description"},
            "hazards": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any hazards detected",
            },
        },
        "required": ["objects", "layout", "hazards"],
        "additionalProperties": False,
    },
}

t5 = time.perf_counter()
try:
    structured_resp = client._client.chat.completions.create(
        model="gemma-4-31b",
        messages=[
            {"role": "system", "content": "Analyze the scene and return valid JSON matching the schema."},
            {"role": "user", "content": "Describe a home office with a desk, chair, laptop, and coffee mug."},
        ],
        temperature=0.1,
        max_completion_tokens=512,
        response_format={
            "type": "json_schema",
            "json_schema": test_schema,
        },
    )
    structured_content = structured_resp.choices[0].message.content
    latency_structured = (time.perf_counter() - t5) * 1000
    log(f"Structured output latency: {latency_structured:.1f}ms")
    log(f"Raw content ({len(structured_content)} chars):")
    log(f"  {structured_content[:600]}")

    structured_json = None
    if structured_content:
        try:
            structured_json = json.loads(structured_content)
            log("STRUCTURED: Valid JSON!")
            log(f"  objects: {structured_json.get('objects', 'N/A')}")
            log(f"  layout: {structured_json.get('layout', 'N/A')}")
            log(f"  hazards: {structured_json.get('hazards', 'N/A')}")
            test5_passed = True
        except json.JSONDecodeError as e:
            log(f"STRUCTURED: JSON parse error: {e}")
            test5_passed = False
    else:
        log("STRUCTURED: Empty response!")
        test5_passed = False
except Exception as e:
    latency_structured = (time.perf_counter() - t5) * 1000
    log(f"STRUCTURED: API call failed after {latency_structured:.1f}ms: {e}")
    test5_passed = False

log(f"TEST 5 {'PASSED' if test5_passed else 'FAILED'}")

# ======================================================================
#  TEST 6: Structured outputs WITH image (multimodal + schema)
# ======================================================================
hr("TEST 6: Multimodal structured outputs (image + json_schema)")

log("Sending image + structured outputs...")

t6 = time.perf_counter()
try:
    multimodal_resp = client._client.chat.completions.create(
        model="gemma-4-31b",
        messages=[
            {"role": "system", "content": "Analyze the image and return valid JSON matching the schema."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see in this image? List objects, layout, and hazards."},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            },
        ],
        temperature=0.1,
        max_completion_tokens=512,
        response_format={
            "type": "json_schema",
            "json_schema": test_schema,
        },
    )
    multimodal_content = multimodal_resp.choices[0].message.content
    latency_multimodal = (time.perf_counter() - t6) * 1000
    log(f"Multimodal structured latency: {latency_multimodal:.1f}ms")
    log(f"Raw content ({len(multimodal_content)} chars):")
    log(f"  {multimodal_content[:600]}")

    multimodal_json = None
    if multimodal_content:
        try:
            multimodal_json = json.loads(multimodal_content)
            log("MULTIMODAL+SCHEMA: Valid JSON!")
            log(f"  objects: {multimodal_json.get('objects', 'N/A')}")
            log(f"  layout: {multimodal_json.get('layout', 'N/A')}")
            log(f"  hazards: {multimodal_json.get('hazards', 'N/A')}")
            test6_passed = True
        except json.JSONDecodeError as e:
            log(f"MULTIMODAL+SCHEMA: JSON parse error: {e}")
            # Show raw content for debugging
            log(f"  Raw response repr: {repr(multimodal_content[:200])}")
            test6_passed = False
    else:
        log("MULTIMODAL+SCHEMA: Empty response!")
        test6_passed = False
except Exception as e:
    latency_multimodal = (time.perf_counter() - t6) * 1000
    log(f"MULTIMODAL+SCHEMA: API call failed after {latency_multimodal:.1f}ms: {e}")
    test6_passed = False

log(f"TEST 6 {'PASSED' if test6_passed else 'FAILED'}")

# ======================================================================
#  SUMMARY
# ======================================================================
hr("RESULTS SUMMARY")

summary = f"""# Command Center Router - Live API Test Results

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
**Model:** gemma-4-31b
**API:** Cerebras Inference ({client._client.base_url})

## Test 1: Text-only signal routing
- **Result:** {'PASSED' if test1_passed else 'FAILED'}
- **Total latency:** {latency_text:.1f}ms
- **Router latency:** {result_text.router_latency_ms:.1f}ms
- **Route to:** {[b.value for b in result_text.decision.route_to] if result_text.decision else 'N/A'}
- **Commands:** {len(result_text.commands)}
- **Branch outputs:** {len(result_text.branch_outputs)}

## Test 2: Image signal routing (multimodal + structured outputs)
- **Result:** {'PASSED' if test2_passed else 'FAILED'}
- **Total latency:** {latency_image:.1f}ms
- **Router latency:** {result_image.router_latency_ms:.1f}ms
- **Route to:** {[b.value for b in result_image.decision.route_to] if result_image.decision else 'N/A'}
- **Commands:** {len(result_image.commands)}
- **Branch outputs:** {len(result_image.branch_outputs)}

## Test 3: Parallel branch dispatch
- **Result:** {'PASSED' if test3_passed else 'LOW OVERLAP'}
- **Wall time:** {latency_parallel:.1f}ms
- **Sequential estimate:** {sequential_estimate:.1f}ms
- **Overlap ratio:** {overlap_ratio:.2f}x
- **Branch details:**
"""

for bo in parallel_outputs:
    summary += f"""  - {bo.branch.value}: {len(bo.content)} chars in {bo.latency_ms:.1f}ms
"""

summary += f"""
## Test 4: Raw API (text-only, NO structured outputs)
- **Result:** {'PASSED' if test4_passed else 'FAILED'}

### Raw response text:
```
{raw_result.content[:800]}
```

### JSON parsing:
- Valid JSON: {'Yes' if raw_json else 'No'}
- Parsed fields: {list(raw_json.keys()) if raw_json else 'N/A'}
- Content: {json.dumps(raw_json, indent=2)[:500] if raw_json else 'N/A'}
"""

summary += f"""
## Test 5: Structured outputs (json_schema, text-only)
- **Result:** {'PASSED' if test5_passed else 'FAILED'}
- **Latency:** {latency_structured:.1f}ms

### Schema output:
{json.dumps(structured_json, indent=2) if test5_passed else structured_content[:500]}
"""

summary += f"""
## Test 6: Multimodal + Structured outputs (image + json_schema)
- **Result:** {'PASSED' if test6_passed else 'FAILED'}
- **Latency:** {latency_multimodal:.1f}ms

### Schema output:
{json.dumps(multimodal_json, indent=2) if test6_passed else multimodal_content[:500] if 'multimodal_content' in dir() else 'N/A'}
"""

# Build recommendations
summary += "\n## Recommendations\n\n"

if test1_passed:
    summary += "- Text routing: WORKING. Router correctly parses JSON schema for text signals.\n"
else:
    summary += "- Text routing: FAILING. Need to debug JSON schema parsing for text signals.\n"

if test2_passed:
    summary += "- Image routing: WORKING. Router handles multimodal input with structured outputs.\n"
else:
    summary += "- Image routing: FAILING.\n"

if test3_passed:
    summary += "- Parallel dispatch: WORKING. Branches execute with overlapping latency.\n"
else:
    summary += "- Parallel dispatch: SEQUENTIAL. ThreadPoolExecutor may not be overlapping.\n"

if test4_passed:
    summary += "- Raw API (text): WORKING. Gemma 4 returns parseable JSON without structured outputs.\n"
else:
    summary += "- Raw API (text): FAILING. Need to examine raw response format.\n"

if test5_passed:
    summary += "- Structured outputs (text): WORKING. json_schema response_format is respected.\n"
else:
    summary += "- Structured outputs (text): FAILING. Cerebras may not support json_schema response_format.\n"

if test6_passed:
    summary += "- Multimodal + structured outputs: WORKING. Images + JSON schema both work.\n"
else:
    summary += "- Multimodal + structured outputs: FAILING. Images with json_schema may not work.\n"

summary += f"""
## Latency Summary
| Test | Latency (ms) |
|------|-------------|
| Text routing | {latency_text:.1f} |
| Image routing | {latency_image:.1f} |
| Parallel dispatch (wall) | {latency_parallel:.1f} |
| Raw text | {latency_raw:.1f} |
| Structured text | {latency_structured:.1f} |
| Multimodal structured | {latency_multimodal:.1f} |
"""

# Write the output
OUTPUT_FILE.write_text(summary)
log(f"\nResults written to {OUTPUT_FILE}")
print("\n" + summary)
