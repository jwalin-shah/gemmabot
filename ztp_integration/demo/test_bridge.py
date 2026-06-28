"""Smoke-test the ZTP bridge (works with or without the native library)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ztp_integration.c_ffi.bridge import ZTPRuntime


def main() -> None:
    ztp = ZTPRuntime()
    mode = "NATIVE" if ztp.available else "MOCK (library not found)"
    print(f"ZTPRuntime initialized — mode: {mode}")
    print()

    # ---- Terran ----
    print("=== Terran — Soil Contact Physics ===")
    for mass in [0.5, 2.5, 10.0]:
        t0 = time.perf_counter()
        r = ztp.terran_evaluate_contact(
            soil_type=1, moisture=0.3, mass_kg=mass,
            footprint_m2=0.01, locomotion=0,
        )
        dt = (time.perf_counter() - t0) * 1000
        print(f"  Mass={mass:5.1f}kg → compaction={r['max_compaction']:.4f}, "
              f"depth={r['compaction_depth_m']:.4f}m, "
              f"yield={r['yield_stress_kpa']:.1f}kPa [{dt:.1f}ms]")
    print()

    # ---- Surgical ----
    print("=== Surgical — Tissue Force Limits ===")
    tissues = [(0, "Liver/Spleen", 1.2), (1, "Bowel/Vessel", 2.5), (2, "Bone/Tendon", 40.0)]
    for tid, tname, max_f in tissues:
        r = ztp.surgical_evaluate_grasp(tissue_type=tid, measured_force_n=max_f * 0.9)
        status = "⚠️  OVERSTRESS" if r["tissue_overstress_detected"] else "✓  Safe"
        print(f"  {tname:16s} ({max_f:.1f}N max) → 90% load: {status}")
    print()

    # ---- Micro ----
    print("=== Micro — Capillary Stiction ===")
    for charge in [50, 120, 200]:
        r = ztp.micro_evaluate_release(electrostatic_charge_v=charge)
        status = "SAFE" if r["safe_to_retract"] else "⚠️  HOLD"
        esd = "ESD!" if r["electrostatic_charge_violation"] else "OK"
        print(f"  Charge={charge:3.0f}V → {status}, ESD:{esd}, "
              f"stiction:{r['release_stiction_active']}")
    print()

    # ---- Atheric ----
    print("=== Atheric — RF Link Quality ===")
    for dist in [0.1, 1.0, 10.0]:
        r = ztp.atheric_handshake(strength=1.0, distance_km=dist)
        mark = "✓" if r["success"] else "✗"
        print(f"  Distance={dist:5.1f}km → {mark} SNR={r['avg_snr_db']:.1f}dB, "
              f"resonance={r['resonance']:.4f}")
    print()

    # ---- Orbital ----
    print("=== Orbital — 6DOF Dynamics (1000 steps) ===")
    state = None
    t0 = time.perf_counter()
    for _ in range(1000):
        state = ztp.orbital_step_6dof(state, dt=0.001)
    dt = (time.perf_counter() - t0) * 1000
    pos = state["position"]
    print(f"  1000 steps in {dt:.1f}ms ({1000/dt:.0f}Hz)")
    print(f"  Final position: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}) m")
    print()

    print("=== All ZTP bridge tests passed ===")


if __name__ == "__main__":
    main()
