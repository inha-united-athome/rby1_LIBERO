"""
RBY1 parallel-jaw gripper with two slide joints.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class RBY1GripperBase(GripperModel):
    """
    RBY1 parallel-jaw gripper (2 slide joints, symmetric).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/rby1_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0])

    @property
    def contact_geom_rgba(self):
        return [0, 0, 0, 0]

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "finger_1_col_0",
                "finger_1_col_1",
                "finger_1_col_2",
                "finger_1_col_3",
                "finger_1_col_4",
            ],
            "right_finger": [
                "finger_2_col_0",
                "finger_2_col_1",
                "finger_2_col_2",
                "finger_2_col_3",
                "finger_2_col_4",
            ],
            "left_fingerpad": ["finger_1_col_0", "finger_1_col_1"],
            "right_fingerpad": ["finger_2_col_0", "finger_2_col_1"],
        }


class RBY1Gripper(RBY1GripperBase):
    """
    1-DoF variant of RBY1GripperBase (single action controls both fingers symmetrically).
    """

    def format_action(self, action):
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(action),
            -1.0,
            1.0,
        )
        return self.current_action

    @property
    def speed(self):
        return 0.20

    @property
    def dof(self):
        return 1
