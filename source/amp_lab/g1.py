"""Provisional G1 profile. Final releases use values exported from the live training environment."""

from .spec import DeploymentSpec, ObservationTerm

JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def default_spec():
    rest, kp, kd, effort, armature = [], [], [], [], []
    for name in JOINT_NAMES:
        a, limit = 0.003609725, 25.0
        if any(x in name for x in ("hip_pitch", "hip_yaw", "waist_yaw")):
            a, limit = 0.010177520, 88.0
        elif "hip_roll" in name or "knee" in name:
            a, limit = 0.025101925, 139.0
        elif "ankle" in name or "waist_" in name:
            a, limit = 2 * 0.003609725, 50.0
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            a, limit = 0.00425, 5.0
        position = 0.0
        for suffix, value in [("hip_pitch", -0.312), ("knee", 0.669), ("ankle_pitch", -0.363),
                              ("elbow", 0.6), ("shoulder_pitch", 0.2)]:
            if suffix in name:
                position = value
        if "shoulder_roll" in name:
            position = 0.2 if name.startswith("left") else -0.2
        frequency = 10 * 2 * 3.1415926535
        rest.append(position)
        kp.append(a * frequency**2)
        kd.append(4 * a * frequency)
        effort.append(limit)
        armature.append(a)
    return DeploymentSpec(
        joint_names=JOINT_NAMES.copy(), default_joint_pos=rest, stiffness=kp, damping=kd,
        effort_limit=effort, action_scale=[0.25] * 29, armature=armature,
        model_xml="robot/mjcf/scene_29dof.xml",
        observations=[ObservationTerm("base_ang_vel", 3, 0.2, 100),
                      ObservationTerm("projected_gravity", 3), ObservationTerm("velocity_commands", 3),
                      ObservationTerm("joint_pos_rel", 29, 1.0, 100),
                      ObservationTerm("joint_vel", 29, 0.05, 100),
                      ObservationTerm("last_action", 29, 1.0, 12)],
        provenance={"status": "provisional_profile_not_a_trained_policy"},
    ).validate()
