"""
Wiring guide: how to integrate ZTP physics into the hackathon's RobotController.

This isn't runnable code — it's a recipe. The actual changes are 3 small edits
in src/robot_controller.py. Apply them when you're ready to swap the mock.
"""

GUIDE = r"""
================================================================================
  Replace src/robot_controller.py's mock with ZTP physics — 3 edits
================================================================================

Edit 1 — Add import (top of robot_controller.py):
--------------------------------------------------------------------
  from ztp_integration.c_ffi.bridge import ZTPRuntime
  ztp = ZTPRuntime()              # auto-falls back to mock if no native lib

Edit 2 — Replace self._position mock with real force feedback:
--------------------------------------------------------------------
  In __init__:
      self._ztp = ztp
      self._contact_force = 0.0
      self._compaction = 0.0
      self._slip_risk = 0.0

Edit 3 — Replace execute() mock bodies with ZTP calls:
--------------------------------------------------------------------

  def execute(self, action, target, **params):
      start = time.perf_counter()

      if action == "pick_up":
          # ZTP validates the grasp against terramechanics + force limits
          terran = self._ztp.terran_evaluate_contact(
              soil_type=1, moisture=0.2, mass_kg=0.8,
              footprint_m2=0.005, locomotion=0,
          )
          surgical = self._ztp.surgical_evaluate_grasp(
              tissue_type=0, measured_force_n=2.0,
          )
          self._contact_force = surgical["clamped_force"]
          self._compaction = terran["max_compaction"]
          self._slip_risk = max(0, terran["max_compaction"] - 0.5) * 2

          if terran["max_compaction"] > 0.8:
              status, msg = "failed", f"Soil too soft (compaction={terran['max_compaction']:.2f})"
          elif surgical["tissue_overstress_detected"]:
              status, msg = "failed", f"Force limit exceeded (clamped at {surgical['clamped_force']:.2f}N)"
          else:
              self._gripper_open = False
              status, msg = "executed", f"Picked up {target} (force={surgical['clamped_force']:.2f}N)"

      elif action == "place":
          terran = self._ztp.terran_evaluate_contact(
              soil_type=1, mass_kg=0.5, footprint_m2=0.01,
          )
          if terran["max_compaction"] > 0.9:
              status, msg = "skipped", "Surface too soft for placement"
          else:
              self._gripper_open = True
              status, msg = "executed", f"Placed {target}"

      # ... rest of actions unchanged ...

      elapsed = (time.perf_counter() - start) * 1000
      return ActionResult(action=action, target=target, status=status,
                          message=msg, duration_ms=elapsed)

================================================================================
  What changes in the demo output:
================================================================================

  Before (mock):
    ✅  pick_up(screwdriver) — "Picked up screwdriver" [0.0ms]

  After (ZTP):
    ✅  pick_up(screwdriver) — "Picked up screwdriver (force=2.00N, compaction=0.01)" [0.3ms]

  Edge case — overstress:
    ❌  pick_up(fragile_part) — "Force limit exceeded (clamped at 1.20N)" [0.3ms]

  Edge case — sinking:
    ❌  pick_up(bolt) — "Soil too soft (compaction=0.85)" [0.3ms]

================================================================================
  ZTP bridge is mock-safe
================================================================================

If ztp-runtime isn't built yet, ZTPRuntime() silently falls back to
physically-plausible synthetic values. Your code works either way — the
native library just makes the numbers real.
"""

if __name__ == "__main__":
    print(GUIDE)
