from dataclasses import dataclass

from typing_extensions import List

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, VerticalAlignment, Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.language import SequentialPlan
from pycram.parameter_inference import (
    ParameterInferenceRule,
    ParameterIdentifier,
    T,
)
from pycram.parameter_rules.default_type_rules import EnumDomain, GraspDomain
from pycram.robot_plans import PickUpActionDescription


@dataclass
class DummyRule(ParameterInferenceRule):

    n_th: int

    def _apply(self, domain: List, context: Context) -> List:
        return domain[: self.n_th]


@dataclass
class EffectRule(ParameterInferenceRule):

    def _apply(self, domain: List[T], context: Context) -> List[T]:
        return [Arms.BOTH, Arms.RIGHT]

    def effect(self):
        new_rule = DummyRule(
            self.parameter_type, self.action_description, self.parameter_name, 1
        )
        self.parameter_infeerer.add_rule(new_rule)


def test_infer_enum_domain(immutable_model_world):

    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomain(Arms), GraspDomain(GraspDescription, robot_view.left_arm.manipulator)
    )

    assert plan.parameter_infeerer.get_domain_for_type(Arms) == [
        Arms.LEFT,
        Arms.RIGHT,
        Arms.BOTH,
    ]

    grasp_domain = plan.parameter_infeerer.get_domain_for_type(GraspDescription)

    assert len(grasp_domain) == 12


def test_infer_domain_with_rules(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomain(Arms), GraspDomain(GraspDescription, robot_view.left_arm.manipulator)
    )

    plan.parameter_infeerer.add_rule(DummyRule(Arms, pick_action, "arm", 1))

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 1
    assert arm_domain[0] == Arms.LEFT


def test_rule_effect(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomain(Arms), GraspDomain(GraspDescription, robot_view.left_arm.manipulator)
    )

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 3
    assert arm_domain == [Arms.LEFT, Arms.RIGHT, Arms.BOTH]

    plan.parameter_infeerer.add_rule(EffectRule(Arms, pick_action, "arm"))

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 2
    assert arm_domain == [Arms.BOTH, Arms.RIGHT]
    assert len(plan.parameter_infeerer.parameter_rules) == 2

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 1
    assert arm_domain == [Arms.BOTH]
    assert len(plan.parameter_infeerer.parameter_rules) == 3


def test_empty_domain(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    arm_type_domain = plan.parameter_infeerer.get_domain_for_type(Arms)

    assert len(arm_type_domain) == 0
    assert arm_type_domain == []

    arm_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "arm")
    )

    assert len(arm_domain) == 0
    assert arm_domain == []


def test_no_body_domain(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pick_action = PickUpActionDescription(
        world.get_body_by_name("milk.stl"),
        ...,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot_view.right_arm.manipulator,
        ),
    )

    plan = SequentialPlan(context, pick_action)

    plan.parameter_infeerer.add_domains(
        EnumDomain(Arms),
        GraspDomain(GraspDescription, robot_view.right_arm.manipulator),
    )

    body_domain = plan.parameter_infeerer.infer_domain_for_parameter(
        ParameterIdentifier(pick_action, "object_designator")
    )

    assert len(body_domain) == 1
    assert body_domain == [world.get_body_by_name("milk.stl")]
