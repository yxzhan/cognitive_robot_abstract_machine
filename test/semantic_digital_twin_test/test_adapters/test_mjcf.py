import os.path

import numpy as np
import pytest

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)

MJCF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)


@pytest.fixture
def table_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "table.xml"))


@pytest.fixture
def kitchen_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "kitchen-small.xml"))


@pytest.fixture
def apartment_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "iai_apartment.xml"))


@pytest.fixture
def pr2_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "pr2_kinematic_tree.xml"))


def test_table_parsing(table_xml_parser):
    body_num = 7
    world = table_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) == body_num

    origin_left_front_leg_joint = world.get_connection(
        world.root, world.kinematic_structure_entities[1]
    )
    assert isinstance(origin_left_front_leg_joint, FixedConnection)


def test_kitchen_parsing(kitchen_xml_parser):
    world = kitchen_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_apartment_parsing(apartment_xml_parser):
    world = apartment_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_pr2_parsing(pr2_xml_parser):
    world = pr2_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0
    assert world.root.name.name == "world"


def test_frame_transform_composes_for_geoms_and_bodies():
    """A <frame> transform must compose into geom origins and parent->child
    connections. Here the frame offsets +y by 2 and rotates +90deg yaw about
    z, so the leaf origin is pos=(0,2,0), rot=Rz(90deg)."""
    xml = """<?xml version='1.0'?>
<mujoco model='frame_test'>
  <worldbody>
    <body name='parent'>
      <frame pos='0.0 2.0 0.0' quat='0.707106781 0 0 0.707106781'>
        <geom name='g' type='box' size='0.1 0.1 0.1'/>
        <body name='child'>
          <joint name='j' type='hinge' axis='0 0 1'/>
          <geom name='g2' type='box' size='0.1 0.1 0.1'/>
        </body>
      </frame>
    </body>
  </worldbody>
</mujoco>"""
    world = MJCFParser.from_xml_string(xml).parse()
    parent = world.get_body_by_name("parent")
    child = world.get_body_by_name("child")

    expected_pos = [0.0, 2.0, 0.0]
    # SDT Quaternion stores (x, y, z, w); 90 deg yaw about z
    expected_quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])

    origin = parent.visual.shapes[0].origin
    geom_quat = origin.to_quaternion().to_np().squeeze()
    np.testing.assert_allclose(
        origin.to_position().to_np().squeeze()[:3], expected_pos, atol=1e-6
    )
    # quaternion sign ambiguity: q and -q represent the same rotation
    assert np.allclose(geom_quat, expected_quat, atol=1e-5) or np.allclose(
        geom_quat, -expected_quat, atol=1e-5
    )

    connection = world.get_connection(parent, child)
    assert isinstance(connection, RevoluteConnection)
    parent_T_joint = connection.parent_T_connection_expression.to_np()
    np.testing.assert_allclose(parent_T_joint[:3, 3].squeeze(), expected_pos, atol=1e-6)
