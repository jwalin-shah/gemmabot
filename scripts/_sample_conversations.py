"""Quick script to extract sample conversations for each chaos type."""
import sys
sys.path.insert(0, ".")

from scripts.exp_stateful_chaos import *
from src.client import CerebrasClient

for chaos_type in ["teleport", "appear", "disappear"]:
    client = CerebrasClient()
    tick_data = []
    print("===== CHAOS TYPE:", chaos_type, "=====")
    for seq_tick in range(8):
        is_post_chaos = seq_tick >= 4
        base_objects = build_objects_from_scene(BASE_SCENE)
        if is_post_chaos:
            if seq_tick == 4:
                chaos_objects, _, render_kwargs, _ = apply_chaos(chaos_type, base_objects, BASE_SCENE, seq_tick)
            else:
                chaos_objects, _, render_kwargs, _ = apply_chaos(chaos_type, base_objects, BASE_SCENE, seq_tick)
            current_objects = chaos_objects
        else:
            current_objects = base_objects
            render_kwargs = {}
        image_b64 = render(current_objects, **render_kwargs)
        prompt = "Tick %d. IDENTIFY every object and its zone." % seq_tick
        tick_data.append({"image": image_b64, "objects": current_objects, "prompt": prompt, "response": None, "is_post_chaos": is_post_chaos})
        call_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        window_start = max(0, len(tick_data) - 2)
        for i, td in enumerate(tick_data):
            if i >= window_start:
                content = [{"type": "text", "text": td["prompt"]}, {"type": "image_url", "image_url": {"url": td["image"]}}]
            else:
                content = [{"type": "text", "text": td["prompt"]}]
            call_messages.append({"role": "user", "content": content})
            if td["response"] is not None:
                call_messages.append({"role": "assistant", "content": td["response"]})
        result = client.chat(call_messages, temperature=0.0, max_tokens=500, response_format={"type": "json_schema", "json_schema": IDENTIFY_SCHEMA})
        tick_data[-1]["response"] = result.content
        parsed = json.loads(result.content)
        if seq_tick == 3 or seq_tick == 4:
            label = "POST-CHAOS" if is_post_chaos else "pre-chaos"
            print("  Tick %d (%s):" % (seq_tick, label))
            print("    changes_detected:", parsed.get("changes_detected", ""))
            obs_list = parsed.get("observed_objects", [])
            obj_str = ", ".join(["%s %s in Zone %s" % (o["color"], o["shape"], o["zone"]) for o in obs_list])
            print("    objects:", obj_str)
        if seq_tick == 7:
            print("  [Last tick] total_objects:", parsed.get("total_objects_visible", "?"))
    print()
print("Done.")
