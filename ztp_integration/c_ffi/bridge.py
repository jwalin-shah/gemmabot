"""
Python ctypes bridge to the John Kruze ztp-runtime shared library.

Provides clean Pythonic wrappers around the C FFI functions exposed by
``libztp_runtime.dylib`` (macOS) / ``libztp_runtime.so`` (Linux).

Usage:
    from ztp_integration.c_ffi.bridge import ZTPRuntime

    ztp = ZTPRuntime()          # auto-loads the library
    result = ztp.terran_evaluate_contact(
        soil_type=1, moisture=0.3, mass_kg=2.5,
        footprint_m2=0.01, locomotion=0,
    )
    print(f"Max compaction: {result['max_compaction']:.4f}")

If the library isn't built yet, ZTPRuntime falls back to a mock that returns
physically-plausible synthetic values so you can develop the integration
before building the Rust kernel.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths to search for the compiled shared library
# ---------------------------------------------------------------------------

_LIB_NAME_MACOS = "libztp_runtime.dylib"
_LIB_NAME_LINUX = "libztp_runtime.so"
_LIB_NAME_WIN = "ztp_runtime.dll"

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_VENDOR_DIR = _PROJECT_ROOT / "ztp_integration" / "vendor" / "ztp-runtime"
_TYPICAL_BUILD_PATHS = [
    _VENDOR_DIR / "target" / "release" / _LIB_NAME_MACOS,
    _VENDOR_DIR / "target" / "release" / _LIB_NAME_LINUX,
    _VENDOR_DIR / "target" / "debug" / _LIB_NAME_MACOS,
    _VENDOR_DIR / "target" / "debug" / _LIB_NAME_LINUX,
    Path("target") / "release" / _LIB_NAME_MACOS,
    Path("target") / "release" / _LIB_NAME_LINUX,
]


def _find_library() -> str | None:
    """Locate the compiled ztp-runtime shared library."""
    # 1. Check ZTP_LIB_PATH env override
    env_path = os.environ.get("ZTP_LIB_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    # 2. Check typical build locations
    for p in _TYPICAL_BUILD_PATHS:
        if p.exists():
            return str(p.resolve())

    # 3. System library path
    found = ctypes.util.find_library("ztp_runtime")
    if found:
        return found

    return None


# ---------------------------------------------------------------------------
# C struct definitions (mirror the Rust FFI exactly)
# ---------------------------------------------------------------------------

class C_SoilResult(ctypes.Structure):
    _fields_ = [
        ("max_compaction", ctypes.c_double),
        ("compaction_depth_m", ctypes.c_double),
    ]


class C_SatelliteState(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_double * 3),
        ("velocity", ctypes.c_double * 3),
        ("quaternion_attitude", ctypes.c_double * 4),
        ("angular_velocity", ctypes.c_double * 3),
        ("inertia_tensor", ctypes.c_double * 9),
    ]


class C_HandshakeResult(ctypes.Structure):
    _fields_ = [
        ("success", ctypes.c_bool),
        ("resonance", ctypes.c_double),
        ("avg_snr_db", ctypes.c_double),
    ]


class C_SurgicalTissueAuditor(ctypes.Structure):
    _fields_ = [
        ("tissue_type_id", ctypes.c_uint32),
        ("max_tearing_force_n", ctypes.c_float),
        ("measured_displacement_m", ctypes.c_float),
        ("measured_force_n", ctypes.c_float),
        ("relaxation_tau", ctypes.c_float),
        ("last_displacement_m", ctypes.c_float),
        ("last_force_n", ctypes.c_float),
        ("accumulated_energy_j", ctypes.c_float),
    ]


class C_SurgicalResult(ctypes.Structure):
    _fields_ = [
        ("tissue_overstress_detected", ctypes.c_bool),
        ("viscoelastic_rupture_detected", ctypes.c_bool),
        ("cable_slip_fault", ctypes.c_bool),
        ("clamped_force", ctypes.c_float),
    ]


class C_MicroReleaseAuditor(ctypes.Structure):
    _fields_ = [
        ("part_mass_micrograms", ctypes.c_float),
        ("pull_off_force_un", ctypes.c_float),
        ("jaw_separation_um", ctypes.c_float),
        ("dynamic_electrostatic_charge_v", ctypes.c_float),
        ("last_jaw_separation_um", ctypes.c_float),
    ]


class C_MicroResult(ctypes.Structure):
    _fields_ = [
        ("release_stiction_active", ctypes.c_bool),
        ("electrostatic_charge_violation", ctypes.c_bool),
        ("piezo_shake_trigger", ctypes.c_bool),
        ("safe_to_retract", ctypes.c_bool),
    ]


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class ZTPRuntime:
    """Python interface to the John Kruze ztp-runtime physics kernel.

    If the native library is not available, a mock backend is used that returns
    physically-plausible synthetic data (deterministic, seeded by inputs).

    Attributes:
        available: True if the native shared library was loaded.
        mock: True if using the synthetic fallback.
    """

    def __init__(self, lib_path: str | None = None) -> None:
        self.available = False
        self.mock = False

        if lib_path:
            path = lib_path
        else:
            found = _find_library()
            path = found

        if path and Path(path).exists():
            self._lib = ctypes.cdll.LoadLibrary(path)
            self._bind_functions()
            self.available = True
        else:
            self._lib = None
            self.mock = True

        self._rng = random.Random(42)

    # ---- public API -------------------------------------------------------

    def terran_evaluate_contact(
        self,
        soil_type: int = 1,
        moisture: float = 0.3,
        glomalin_mg_g: float = 1.0,
        compaction: float = 0.5,
        depth_layers: int = 10,
        mass_kg: float = 2.5,
        footprint_m2: float = 0.01,
        locomotion: int = 0,
    ) -> dict[str, float]:
        """Evaluate soil stress from a robot contact footprint.

        Args:
            soil_type: 0=sand, 1=loam, 2=clay, 3=silt
            moisture: volumetric water content (0-1)
            glomalin_mg_g: mycorrhizal glycoprotein content
            compaction: baseline bulk density fraction
            depth_layers: number of depth integration layers
            mass_kg: robot/end-effector mass
            footprint_m2: contact patch area
            locomotion: 0=static, 1=wheeled, 2=legged, 3=tracked

        Returns:
            dict with keys: max_compaction, compaction_depth_m, yield_stress_kpa,
            safety_factor
        """
        if self.available:
            self._lib.ztp_terran_evaluate_contact.restype = C_SoilResult
            result = self._lib.ztp_terran_evaluate_contact(
                ctypes.c_int(soil_type),
                ctypes.c_double(moisture),
                ctypes.c_double(glomalin_mg_g),
                ctypes.c_double(compaction),
                ctypes.c_uint(depth_layers),
                ctypes.c_double(mass_kg),
                ctypes.c_double(footprint_m2),
                ctypes.c_int(locomotion),
            )
            return {
                "max_compaction": result.max_compaction,
                "compaction_depth_m": result.compaction_depth_m,
                "yield_stress_kpa": self._estimate_yield_stress(soil_type, moisture, glomalin_mg_g),
                "safety_factor": 1.0 - result.max_compaction,
            }
        return self._mock_terran(soil_type, moisture, mass_kg, footprint_m2, locomotion)

    def orbital_step_6dof(
        self,
        state: dict[str, list[float]] | None = None,
        dt: float = 0.001,
    ) -> dict[str, list[float]]:
        """Step 6DOF orbital dynamics forward by dt seconds.

        Args:
            state: optional initial state dict with keys: position[3],
                   velocity[3], quaternion_attitude[4], angular_velocity[3],
                   inertia_tensor[9]. Default is LEO test orbit.
            dt: time step in seconds (default 1ms = 1000Hz)

        Returns:
            Updated state dict (in-place semantics preserved)
        """
        if state is None:
            state = self._default_orbital_state()

        if self.available:
            c_state = self._dict_to_c_state(state)
            self._lib.ztp_orbital_step_6dof(ctypes.byref(c_state), ctypes.c_double(dt))
            return self._c_state_to_dict(c_state)
        return self._mock_orbital_step(state, dt)

    def orbital_step_attitude(
        self,
        state: dict[str, list[float]] | None = None,
        torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = 0.001,
    ) -> dict[str, list[float]]:
        """Step attitude dynamics only (no translation).

        Args:
            state: orbital state dict (only attitude fields used)
            torque: (x, y, z) external torque in Nm
            dt: time step in seconds

        Returns:
            Updated state dict
        """
        if state is None:
            state = self._default_orbital_state()

        if self.available:
            c_state = self._dict_to_c_state(state)
            self._lib.ztp_orbital_step_attitude(
                ctypes.byref(c_state),
                ctypes.c_double(torque[0]),
                ctypes.c_double(torque[1]),
                ctypes.c_double(torque[2]),
                ctypes.c_double(dt),
            )
            return self._c_state_to_dict(c_state)
        return self._mock_orbital_attitude(state, torque, dt)

    def atheric_handshake(
        self,
        seed: bytes | None = None,
        strength: float = 1.0,
        distance_km: float = 1.0,
    ) -> dict[str, float | bool]:
        """Evaluate RF link quality under cryptographic channel hopping.

        Args:
            seed: 32-byte SHA-256 seed for frequency hopping
            strength: transmitter power in arbitrary units
            distance_km: link distance in km

        Returns:
            dict with keys: success, resonance, avg_snr_db
        """
        if seed is None:
            seed = bytes(range(32))
        if len(seed) != 32:
            raise ValueError("seed must be exactly 32 bytes (SHA-256 output)")

        if self.available:
            self._lib.ztp_atheric_handshake.restype = C_HandshakeResult
            result = self._lib.ztp_atheric_handshake(
                ctypes.c_char_p(seed),
                ctypes.c_double(strength),
                ctypes.c_double(distance_km),
            )
            return {
                "success": bool(result.success),
                "resonance": result.resonance,
                "avg_snr_db": result.avg_snr_db,
            }
        return self._mock_atheric(strength, distance_km)

    def surgical_evaluate_grasp(
        self,
        tissue_type: int = 0,
        measured_force_n: float = 0.5,
        measured_displacement_m: float = 0.001,
        dt: float = 0.001,
    ) -> dict[str, Any]:
        """Evaluate surgical grasp safety with tissue force limits.

        Args:
            tissue_type: 0=Liver/Spleen (1.2N max), 1=Bowel/Vessel (2.5N max),
                         2=Bone/Tendon (40N max)
            measured_force_n: current force reading in Newtons
            measured_displacement_m: current displacement in meters
            dt: time step in seconds

        Returns:
            dict with keys: tissue_overstress_detected, viscoelastic_rupture_detected,
            cable_slip_fault, clamped_force, max_allowed_force
        """
        max_forces = {0: 1.2, 1: 2.5, 2: 40.0}
        max_force = max_forces.get(tissue_type, 1.2)

        if self.available:
            auditor = C_SurgicalTissueAuditor(
                tissue_type_id=tissue_type,
                max_tearing_force_n=max_force,
                measured_displacement_m=measured_displacement_m,
                measured_force_n=measured_force_n,
                relaxation_tau=0.05,
                last_displacement_m=measured_displacement_m * 0.9,
                last_force_n=measured_force_n * 0.85,
                accumulated_energy_j=measured_force_n * measured_displacement_m * 0.5,
            )
            self._lib.ztp_surgical_evaluate_grasp.restype = C_SurgicalResult
            result = self._lib.ztp_surgical_evaluate_grasp(
                ctypes.byref(auditor),
                ctypes.c_float(dt),
            )
            return {
                "tissue_overstress_detected": bool(result.tissue_overstress_detected),
                "viscoelastic_rupture_detected": bool(result.viscoelastic_rupture_detected),
                "cable_slip_fault": bool(result.cable_slip_fault),
                "clamped_force": result.clamped_force,
                "max_allowed_force": max_force,
            }
        return self._mock_surgical(tissue_type, measured_force_n, max_force)

    def micro_evaluate_release(
        self,
        part_mass_micrograms: float = 50.0,
        pull_off_force_un: float = 120.0,
        jaw_separation_um: float = 10.0,
        electrostatic_charge_v: float = 50.0,
        dt: float = 0.001,
    ) -> dict[str, bool]:
        """Evaluate micro-part release conditions (capillary stiction, ESD).

        Args:
            part_mass_micrograms: part mass in micrograms
            pull_off_force_un: capillary stiction tension in micronewtons
            jaw_separation_um: jaw opening in micrometers
            electrostatic_charge_v: surface charge in volts
            dt: time step in seconds

        Returns:
            dict with keys: release_stiction_active, electrostatic_charge_violation,
            piezo_shake_trigger, safe_to_retract
        """
        if self.available:
            auditor = C_MicroReleaseAuditor(
                part_mass_micrograms=part_mass_micrograms,
                pull_off_force_un=pull_off_force_un,
                jaw_separation_um=jaw_separation_um,
                dynamic_electrostatic_charge_v=electrostatic_charge_v,
                last_jaw_separation_um=jaw_separation_um * 0.9,
            )
            self._lib.ztp_micro_evaluate_release.restype = C_MicroResult
            result = self._lib.ztp_micro_evaluate_release(
                ctypes.byref(auditor),
                ctypes.c_float(dt),
            )
            return {
                "release_stiction_active": bool(result.release_stiction_active),
                "electrostatic_charge_violation": bool(result.electrostatic_charge_violation),
                "piezo_shake_trigger": bool(result.piezo_shake_trigger),
                "safe_to_retract": bool(result.safe_to_retract),
            }
        return self._mock_micro(electrostatic_charge_v)

    # ---- FFI binding ------------------------------------------------------

    def _bind_functions(self) -> None:
        lib = self._lib

        # Terran
        lib.ztp_terran_evaluate_contact.argtypes = [
            ctypes.c_int, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_uint, ctypes.c_double,
            ctypes.c_double, ctypes.c_int,
        ]
        lib.ztp_terran_evaluate_contact.restype = C_SoilResult

        # Orbital
        lib.ztp_orbital_step_6dof.argtypes = [ctypes.POINTER(C_SatelliteState), ctypes.c_double]
        lib.ztp_orbital_step_6dof.restype = None

        lib.ztp_orbital_step_attitude.argtypes = [
            ctypes.POINTER(C_SatelliteState),
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double,
        ]
        lib.ztp_orbital_step_attitude.restype = None

        # Atheric
        lib.ztp_atheric_handshake.argtypes = [
            ctypes.c_char_p, ctypes.c_double, ctypes.c_double,
        ]
        lib.ztp_atheric_handshake.restype = C_HandshakeResult

        # Surgical
        lib.ztp_surgical_evaluate_grasp.argtypes = [
            ctypes.POINTER(C_SurgicalTissueAuditor), ctypes.c_float,
        ]
        lib.ztp_surgical_evaluate_grasp.restype = C_SurgicalResult

        # Micro
        lib.ztp_micro_evaluate_release.argtypes = [
            ctypes.POINTER(C_MicroReleaseAuditor), ctypes.c_float,
        ]
        lib.ztp_micro_evaluate_release.restype = C_MicroResult

    # ---- state helpers ----------------------------------------------------

    @staticmethod
    def _default_orbital_state() -> dict[str, list[float]]:
        """LEO test orbit: ~400km altitude, circular."""
        return {
            "position": [6_771_000.0, 0.0, 0.0],           # m (Earth radius + 400km)
            "velocity": [0.0, 7_667.0, 0.0],                 # m/s (circular orbit)
            "quaternion_attitude": [1.0, 0.0, 0.0, 0.0],    # identity
            "angular_velocity": [0.0, 0.0, 0.001],           # rad/s
            "inertia_tensor": [100, 0, 0, 0, 100, 0, 0, 0, 100],
        }

    @staticmethod
    def _dict_to_c_state(d: dict) -> C_SatelliteState:
        s = C_SatelliteState()
        for i in range(3):
            s.position[i] = d["position"][i]
            s.velocity[i] = d["velocity"][i]
            s.angular_velocity[i] = d["angular_velocity"][i]
        for i in range(4):
            s.quaternion_attitude[i] = d["quaternion_attitude"][i]
        for i in range(9):
            s.inertia_tensor[i] = d["inertia_tensor"][i]
        return s

    @staticmethod
    def _c_state_to_dict(s: C_SatelliteState) -> dict[str, list[float]]:
        return {
            "position": [s.position[i] for i in range(3)],
            "velocity": [s.velocity[i] for i in range(3)],
            "quaternion_attitude": [s.quaternion_attitude[i] for i in range(4)],
            "angular_velocity": [s.angular_velocity[i] for i in range(3)],
            "inertia_tensor": [s.inertia_tensor[i] for i in range(9)],
        }

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _estimate_yield_stress(soil_type: int, moisture: float, glomalin: float) -> float:
        base = [50, 80, 120, 60][soil_type % 4]
        g_factor = 1.0 + glomalin * 0.3
        m_factor = max(0.3, 1.0 - moisture * 0.5)
        return base * g_factor * m_factor

    # ---- mock fallbacks (deterministic, physically-plausible) -------------

    def _mock_terran(
        self, soil_type: int, moisture: float, mass_kg: float,
        footprint_m2: float, locomotion: int,
    ) -> dict[str, float]:
        # Boussinesq half-space approximation
        pressure = (mass_kg * 9.81) / max(footprint_m2, 0.001) / 1000  # kPa
        compaction = min(1.0, pressure * 0.001 * (1.0 + moisture * 0.5))
        depth = math.sqrt(footprint_m2 / math.pi) * 0.5 * (1.0 + locomotion * 0.3)
        yield_stress = self._estimate_yield_stress(soil_type, moisture, 1.0)
        return {
            "max_compaction": round(compaction, 6),
            "compaction_depth_m": round(depth, 6),
            "yield_stress_kpa": round(yield_stress, 2),
            "safety_factor": round(1.0 - compaction, 6),
        }

    def _mock_orbital_step(self, state: dict, dt: float) -> dict[str, list[float]]:
        # Simple 2-body Kepler step (no J2 or relativistic corrections)
        mu = 3.986004418e14
        pos = state["position"]
        vel = state["velocity"]
        r = math.hypot(*pos)
        a = -mu / (r ** 3)
        return {
            "position": [pos[i] + vel[i] * dt for i in range(3)],
            "velocity": [vel[i] + a * pos[i] * dt for i in range(3)],
            "quaternion_attitude": state["quaternion_attitude"],
            "angular_velocity": state["angular_velocity"],
            "inertia_tensor": state["inertia_tensor"],
        }

    def _mock_orbital_attitude(
        self, state: dict, torque: tuple, dt: float,
    ) -> dict[str, list[float]]:
        # Simple Euler integration of angular velocity
        I = 100.0  # scalar inertia approx
        av = state["angular_velocity"]
        return {
            **state,
            "angular_velocity": [
                av[0] + torque[0] / I * dt,
                av[1] + torque[1] / I * dt,
                av[2] + torque[2] / I * dt,
            ],
        }

    def _mock_atheric(self, strength: float, distance_km: float) -> dict[str, float | bool]:
        # Friis path loss + Shannon capacity
        path_loss = (4 * math.pi * distance_km * 1000 / 0.05) ** 2  # 50mm wavelength
        snr = strength / max(path_loss, 1e-10)
        snr_db = 10 * math.log10(max(snr, 1e-10))
        return {
            "success": snr_db > 10,
            "resonance": round(1.0 / (1.0 + path_loss * 1e-12), 6),
            "avg_snr_db": round(snr_db, 2),
        }

    def _mock_surgical(
        self, tissue_type: int, force_n: float, max_force: float,
    ) -> dict[str, Any]:
        overstress = force_n > max_force * 0.85
        rupture = force_n > max_force * 1.1
        return {
            "tissue_overstress_detected": overstress,
            "viscoelastic_rupture_detected": rupture,
            "cable_slip_fault": force_n < 0.01 and not overstress,
            "clamped_force": min(force_n, max_force),
            "max_allowed_force": max_force,
        }

    def _mock_micro(self, charge_v: float) -> dict[str, bool]:
        return {
            "release_stiction_active": True,
            "electrostatic_charge_violation": charge_v > 150.0,
            "piezo_shake_trigger": charge_v > 100.0,
            "safe_to_retract": charge_v < 80.0,
        }
