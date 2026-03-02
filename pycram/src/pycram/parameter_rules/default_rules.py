from dataclasses import dataclass
from typing import List

from krrood.entity_query_language.symbolic import SymbolicExpression
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.parameter_inference import InferenceRule, T, DesignatorDomain
from pycram.view_manager import ViewManager


@dataclass
class ArmsFitGraspDescriptionRule(InferenceRule):

    def rule(self, domain: DesignatorDomain, context: Context) -> SymbolicExpression:
        variables = domain.create_variables()
        arm_var = None
        grasp_var = None
        for name, variable in variables.items():
            if variable._type_ == Arms:
                arm_var = variable
            elif variable._type_ == GraspDescription:
                grasp_var = variable
        return grasp_var.manipulator == ViewManager.get_end_effector_view(
            arm_var, context.robot
        )
