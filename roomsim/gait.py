"""Procedural walk cycle: joint angles from gait phase and walking speed (numpy only)."""
import numpy as np

JOINTS = ["hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R",
          "shoulder_L", "shoulder_R", "elbow_L", "elbow_R", "spine_twist", "pelvis_bob"]
MAX_SPEED = 1.5  # m/s, matches env


def cadence_hz(speed):
    """Full gait cycles per second (two steps per cycle). ~1.05 Hz at 1.3 m/s."""
    return 0.5 + 0.42 * np.clip(np.abs(speed), 0.0, MAX_SPEED)


def advance_phase(phase, speed, dt):
    return np.mod(phase + 2 * np.pi * cadence_hz(speed) * dt, 2 * np.pi)


def _leg(phi, s):
    hip = 0.42 * s * np.sin(phi)
    swing = np.maximum(0.0, np.cos(phi))  # leg moving forward
    knee = s * (0.10 + 0.85 * swing ** 1.5)
    ankle = s * (0.18 * swing - 0.12 * np.maximum(0.0, -np.cos(phi)))
    return hip, knee, ankle


def joint_angles(phase, speed):
    """phase, speed: arrays of equal shape -> array [..., len(JOINTS)] (radians / metres for bob)."""
    phase = np.asarray(phase, dtype=np.float64)
    s = np.clip(np.abs(np.asarray(speed, dtype=np.float64)) / 1.3, 0.0, 1.15)
    hL, kL, aL = _leg(phase, s)
    hR, kR, aR = _leg(phase + np.pi, s)
    shL, shR = -0.38 * s * np.sin(phase), -0.38 * s * np.sin(phase + np.pi)
    elL = 0.15 + 0.30 * s * np.maximum(0.0, np.sin(phase + np.pi))
    elR = 0.15 + 0.30 * s * np.maximum(0.0, np.sin(phase))
    twist = 0.07 * s * np.sin(phase)
    bob = 0.025 * s * np.cos(2 * phase)
    return np.stack([hL, hR, kL, kR, aL, aR, shL, shR, elL, elR, twist, bob], axis=-1)
