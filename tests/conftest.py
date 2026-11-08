from pathlib import Path

import pytest
from amp_lab.spec import DeploymentSpec, ObservationTerm


@pytest.fixture
def tiny_robot(tmp_path):
    xml = tmp_path / "robot" / "scene.xml"
    xml.parent.mkdir()
    xml.write_text('''<mujoco><option timestep="0.005" gravity="0 0 0"/>
      <worldbody><body name="pelvis" pos="0 0 0.8"><freejoint/>
        <geom type="sphere" size="0.05" mass="1"/>
        <body name="first"><joint name="a" axis="0 1 0" range="-170 170"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.2"/>
          <body name="second" pos="0 0 -0.2"><joint name="b" axis="1 0 0" range="-170 170"/>
            <geom type="sphere" size="0.02" mass="0.1"/>
          </body>
        </body>
      </body></worldbody><actuator><motor joint="a"/><motor joint="b"/></actuator></mujoco>''')
    return xml


@pytest.fixture
def tiny_spec():
    # Deliberately reversed from the XML order.
    return DeploymentSpec(
        joint_names=["b", "a"], default_joint_pos=[0.1, -0.2], stiffness=[20, 10], damping=[1, 1],
        effort_limit=[4, 3], action_scale=[0.2, 0.3], armature=[0.01, 0.02],
        observations=[ObservationTerm("base_ang_vel", 3, 0.2, 100),
                      ObservationTerm("projected_gravity", 3), ObservationTerm("velocity_commands", 3),
                      ObservationTerm("joint_pos_rel", 2), ObservationTerm("joint_vel", 2, 0.05, 100),
                      ObservationTerm("last_action", 2)],
    ).validate()


def write_zero_onnx(path: Path, spec):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["observation", "zero"], ["action"])], "untrained_test_fixture",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, [1, spec.observation_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, len(spec.joint_names)])],
        [numpy_helper.from_array(np.zeros((spec.observation_dim, len(spec.joint_names)), np.float32), "zero")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save(model, path)
