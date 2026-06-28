# John Kruze G^G Physics Integration

**Bridge the Cerebras × Gemma 4 multi-agent pipeline with John Kruze's deterministic physics engine for physics-grounded robot control.**

## Why This Exists

The hackathon demo has a `RobotController` with mock timing and a 2D tabletop sim with simple position-based physics. John Kruze's [ztp-runtime](https://github.com/johnkruze/ztp-runtime) provides 1000Hz bare-metal physics solvers for:

- **Terran** — Boussinesq soil mechanics (ground-contact forces, compaction, slip)
- **Surgical** — Tissue force-limit enforcement (45N/2ms reflex)
- **Micro** — Capillary stiction and electrostatic discharge (micro-part handling)
- **Orbital** — 20D relativistic 6DOF dynamics
- **Atheric** — RF coherence and SHA-256 channel hopping

This directory contains everything to build, bridge, and integrate those solvers into the hackathon pipeline — without touching the existing `src/` code.

## Architecture

```
Your hackathon pipeline (src/)
  ├── Vision Agent (Gemma 4 multimodal) → scene understanding
  ├── Action Agent (Gemma 4 text) → action plan
  ├── Safety Agent (Gemma 4 text) → constraint check
  └── Robot Execute ──→ ztp_bridge ──→ libztp_runtime (Rust kernel)
                                           └── 1000Hz physics validation
                                                └── Force/torque feedback
                                                     └── Somatic signature
```

## Quick Start

### 1. Install Rust toolchain (if needed)

```bash
bash ztp_integration/scripts/install_rust.sh
```

### 2. Clone & build ztp-runtime

```bash
bash ztp_integration/scripts/build_ztp.sh
```

This clones the repo and builds the shared library (`libztp_runtime.dylib` on macOS).

### 3. Test the bridge

```bash
cd ztp_integration
uv run python -m demo.test_bridge
```

### 4. Run the integration demo

```bash
cd ztp_integration
uv run python -m demo.integrate_with_sim
```

## Integration Points

### A. Swap `RobotController.execute()` (easiest)

Replace the mock controller with real physics:

```python
from ztp_integration.c_ffi.bridge import ZTPRuntime

ztp = ZTPRuntime()

# Instead of mock "executed" in 0ms:
result = ztp.terran_evaluate_contact(
    soil_type=1, moisture=0.3, mass_kg=2.5,
    footprint_m2=0.01, locomotion=0
)
# Returns real compaction, slip risk, force feedback
```

### B. Augment `src/sim/world.py` physics

Add Terran soil-contact forces and surgical force limits to the tabletop sim's `World.physics()` method so gripper-object interaction has realistic stress/strain.

### C. Full ZTP stabilizer loop

Wire the reactive loop to call ZTP reflex stabilizers (dexterous-hand slip reflex, grounded navigation traction calibration) as a somatic validation layer between Gemma 4's decision and actual actuation.

## Directory Layout

```
ztp_integration/
├── README.md               ← This file
├── c_ffi/
│   ├── __init__.py
│   └── bridge.py           ← Python ctypes bridge to libztp_runtime
├── scripts/
│   ├── install_rust.sh     ← Install Rust toolchain
│   └── build_ztp.sh        ← Clone & build ztp-runtime
├── demo/
│   ├── test_bridge.py      ← Smoke-test the FFI bridge
│   └── integrate_with_sim.py ← Show how to wire into hackathon pipeline
└── vendor/                 ← (auto-created) cloned repos
```

## FFI Functions Available

| Function | Domain | Input | Output |
|----------|--------|-------|--------|
| `ztp_terran_evaluate_contact` | Soil mechanics | soil_type, moisture, glomalin, compaction, depth, mass, footprint, locomotion | max_compaction, compaction_depth_m |
| `ztp_orbital_step_6dof` | Satellite dynamics | state (pos, vel, quat, angvel, inertia), dt | updated state (in-place) |
| `ztp_orbital_step_attitude` | Attitude control | state, ext_torque, dt | updated state (in-place) |
| `ztp_atheric_handshake` | RF coherence | seed (32 bytes), strength, distance_km | success, resonance, avg_snr_db |
| `ztp_surgical_evaluate_grasp` | Tissue force | auditor struct, dt | overstress, rupture, cable_slip, clamped_force |
| `ztp_micro_evaluate_release` | Micro-part handling | auditor struct | stiction_active, esd_violation, shake_trigger, safe_to_retract |

## Demo Video Script Addition (for hackathon submission)

Add this segment to your 60-second demo to highlight the integration:

> **15-20s**: "Cerebras plans the action in 100ms..."
> **20-25s**: "...then John Kruze G^G validates it against physics at 1000Hz — force limits, soil compaction, slip detection."
> **25-30s**: Split screen: mock controller (instant, fake) vs ZTP-validated (real force curves, SHA-256 signatures)

## License

MIT — same as the parent hackathon project. John Kruze repos are MIT/Apache 2.0 dual-licensed.
