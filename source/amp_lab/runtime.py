import numpy as np
from scipy.spatial.transform import Rotation

from .spec import DeploymentSpec


def rotate_inverse(quat_wxyz, vector):
    quat = np.asarray(quat_wxyz, dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply(vector)


class ObservationBuilder:
    def __init__(self, spec: DeploymentSpec):
        self.spec = spec.validate()
        self.history = np.zeros((spec.history_length, spec.frame_dim), dtype=np.float32)
        self.last_action = np.zeros(len(spec.joint_names), dtype=np.float32)

    def reset(self):
        self.history.fill(0)
        self.last_action.fill(0)

    def build(self, *, quat_wxyz, ang_vel_b, joint_pos, joint_vel, command, lin_vel_b=None):
        values = {
            "base_ang_vel": ang_vel_b,
            "projected_gravity": rotate_inverse(quat_wxyz, [0.0, 0.0, -1.0]),
            "velocity_commands": command,
            "joint_pos_rel": np.asarray(joint_pos) - self.spec.default_joint_pos,
            "joint_vel": joint_vel,
            "last_action": self.last_action,
            "base_lin_vel": lin_vel_b,
        }
        terms = []
        for term in self.spec.observations:
            if values[term.name] is None:
                raise ValueError(f"Missing state: {term.name}")
            value = np.asarray(values[term.name], dtype=np.float32)
            if value.shape != (term.size,) or not np.isfinite(value).all():
                raise ValueError(f"Bad shape or non-finite {term.name}: {value.shape}")
            # IsaacLab ObservationManager applies clipping before scaling.
            if term.clip is not None:
                value = np.clip(value, -term.clip, term.clip)
            terms.append(value * term.scale)
        frame = np.concatenate(terms)
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame
        return self.history.reshape(-1).copy()

    def action_target(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != self.last_action.shape or not np.isfinite(action).all():
            raise ValueError("Policy returned wrong action shape or non-finite actions")
        clipped = np.clip(action, -self.spec.action_clip, self.spec.action_clip)
        self.last_action[:] = clipped
        return np.asarray(self.spec.default_joint_pos) + clipped * self.spec.action_scale


class OnnxPolicy:
    def __init__(self, path, spec: DeploymentSpec):
        import onnxruntime as ort

        from .motion import sha256

        self.sha256 = sha256(path)

        # Batch-one actors are small; avoid a CPU thread pool competing with GPU simulation.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Only feed-forward actors with one input/output are supported")
        if inputs[0].type != "tensor(float)" or inputs[0].shape != [1, spec.observation_dim]:
            raise ValueError(f"Policy input mismatch: {inputs[0].type} {inputs[0].shape}")
        if outputs[0].shape != [1, len(spec.joint_names)]:
            raise ValueError(f"Policy output mismatch: {outputs[0].shape}")
        self.input_name = inputs[0].name

    def __call__(self, observation):
        return self.session.run(None, {self.input_name: observation[None].astype(np.float32)})[0][0]
