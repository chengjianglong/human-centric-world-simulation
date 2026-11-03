"""Humanoid skeleton shared by Blender (armature) and BVH export.

Rest pose: standing, facing -Y (Blender front), left side at +X, metres, z up,
feet on z=0. Joint angle conventions (see gait.py): positive hip/shoulder = limb
swings FORWARD, positive knee = shin bends BACK, positive elbow = forearm bends
FORWARD, positive ankle = toes UP.
"""

# name, parent, head(x,y,z), tail(x,y,z), limb radius (for mesh)
BONES = [
    ("pelvis",      None,          (0.0, 0.0, 0.95),  (0.0, 0.0, 1.06),   0.15),
    ("spine",       "pelvis",      (0.0, 0.0, 1.06),  (0.0, 0.0, 1.40),   0.16),
    ("neck",        "spine",       (0.0, 0.0, 1.40),  (0.0, 0.0, 1.53),   0.05),
    ("head",        "neck",        (0.0, 0.0, 1.53),  (0.0, 0.0, 1.76),   0.10),
    ("thigh.L",     "pelvis",      (0.10, 0.0, 0.95), (0.10, 0.0, 0.52),  0.075),
    ("shin.L",      "thigh.L",     (0.10, 0.0, 0.52), (0.10, 0.0, 0.08),  0.06),
    ("foot.L",      "shin.L",      (0.10, 0.0, 0.08), (0.10, -0.16, 0.03), 0.045),
    ("thigh.R",     "pelvis",      (-0.10, 0.0, 0.95), (-0.10, 0.0, 0.52), 0.075),
    ("shin.R",      "thigh.R",     (-0.10, 0.0, 0.52), (-0.10, 0.0, 0.08), 0.06),
    ("foot.R",      "shin.R",      (-0.10, 0.0, 0.08), (-0.10, -0.16, 0.03), 0.045),
    ("upper_arm.L", "spine",       (0.20, 0.0, 1.38), (0.23, 0.0, 1.10),  0.045),
    ("forearm.L",   "upper_arm.L", (0.23, 0.0, 1.10), (0.24, -0.02, 0.84), 0.038),
    ("upper_arm.R", "spine",       (-0.20, 0.0, 1.38), (-0.23, 0.0, 1.10), 0.045),
    ("forearm.R",   "upper_arm.R", (-0.23, 0.0, 1.10), (-0.24, -0.02, 0.84), 0.038),
]

# Which gait joint drives each bone and with what sign about the bone's local X
# axis in Blender (measured: +X on a downward bone swings its tail toward +Y = backward).
BONE_DRIVE = {
    "thigh.L": ("hip_L", -1.0), "thigh.R": ("hip_R", -1.0),
    "shin.L": ("knee_L", 1.0), "shin.R": ("knee_R", 1.0),
    "foot.L": ("ankle_L", -1.0), "foot.R": ("ankle_R", -1.0),
    "upper_arm.L": ("shoulder_L", -1.0), "upper_arm.R": ("shoulder_R", -1.0),
    "forearm.L": ("elbow_L", -1.0), "forearm.R": ("elbow_R", -1.0),
}
# spine twist is about the bone's local Y (its long axis); pelvis bob moves the pelvis along local Y (= up).
