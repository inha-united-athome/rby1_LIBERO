"""
Rainbow Robotics RBY1 v1.1 humanoid robot model for robosuite.

Merged from our multi-variant implementation + rby1_LIBERO optimizations:
  - Ours:   multi-variant support, visual gripper meshes, transparent collision geoms
  - LIBERO: OSC_POSE controller, optimized init_qpos (gripper-down), table offsets

Variants:
  - RBY1:              Full robot (wheels, torso, dual arms, head)
  - RBY1FixedLowerBody: Wheels removed, everything else active
  - RBY1BothArms:      Both arms only (torso/head/wheels fixed)
  - RBY1RightArm:      Right arm only (single arm, OSC_POSE)
  - RBY1LeftArm:       Left arm only (single arm, OSC_POSE)
"""
import numpy as np

from robosuite.models.robots.manipulators.legged_manipulator_model import LeggedManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

# ── Optimized 7-DOF init_qpos from LIBERO ──
# Gripper pointing downward (like Panda home), suitable for table-top manipulation
# Joint order: arm_0(shoulder_yaw) arm_1(shoulder_pitch) arm_2(shoulder_roll)
#              arm_3(elbow) arm_4(wrist_roll) arm_5(wrist_pitch) arm_6(wrist_yaw)
_RIGHT_ARM_QPOS = np.array([-0.35, -0.60, -1.10, -1.00, -3.00, -1.74, -2.00])
_LEFT_ARM_QPOS  = np.array([-0.35,  0.60,  1.10, -1.00,  3.00, -1.74,  2.00])


class RBY1(LeggedManipulatorModel):
    """
    Full RBY1 v1.1 robot with wheels, torso, dual arms, and head.
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/rby1/robot.xml"), idn=idn)
        # Base class does NOT remove static visuals — subclasses handle it
        # based on which arm(s) get grippers mounted.

    def _remove_static_ee_visual(self, side):
        """Remove the static visual EE body + finger bodies for the given side.
        These are only needed on arms that do NOT have a gripper mounted."""
        prefix = self.naming_prefix  # e.g. "robot0_"
        # Collect the raw geom names (without prefix) that will be removed
        removed_geoms_raw = set()
        for suffix in ["ee_visual", "finger1_visual", "finger2_visual"]:
            name = f"{prefix}{side}_{suffix}"
            body = self.worldbody.find(f".//body[@name='{name}']")
            if body is not None:
                for geom in body.findall("geom"):
                    gname = geom.get("name")
                    if gname:
                        # Strip prefix to get raw name (as stored in _visual_geoms)
                        raw = gname[len(prefix):] if gname.startswith(prefix) else gname
                        removed_geoms_raw.add(raw)
                parent = self.worldbody.find(f".//body[@name='{name}']/..")
                if parent is not None:
                    parent.remove(body)
        # Clean up robosuite's internal tracking lists (_visual_geoms stores raw names)
        if removed_geoms_raw:
            self._visual_geoms = [g for g in self._visual_geoms if g not in removed_geoms_raw]
            self._contact_geoms = [g for g in self._contact_geoms if g not in removed_geoms_raw]

    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        return {"right": "RBY1Gripper", "left": "RBY1Gripper"}

    @property
    def default_controller_config(self):
        return {
            "right": "default_rby1",
            "left": "default_rby1",
            "head": "default_rby1_head",
            "torso": "default_rby1_torso",
        }

    @property
    def init_qpos(self):
        # 24 joints total (2 wheels + 6 torso + 7 right arm + 7 left arm + 2 head)
        init_qpos = np.zeros(24)
        init_qpos[8:15] = _RIGHT_ARM_QPOS
        init_qpos[15:22] = _LEFT_ARM_QPOS
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, 0, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.075 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.075 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.075 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.075 - table_length / 2, 0, 0),
            "living_room_table": lambda table_length: (-0.075 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        return {"right": "right_eef", "left": "left_eef"}

    @property
    def contact_geom_rgba(self):
        return [0, 0, 0, 0]

    @property
    def gripper_mount_quat_offset(self):
        return {}


class RBY1FixedLowerBody(RBY1):
    """
    RBY1 with wheels removed (fixed base). Keeps torso, arms, and head.
    """

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("wheel")
        self._remove_free_joint()
        # Both arms get grippers → remove static visuals on both sides
        self._remove_static_ee_visual("right")
        self._remove_static_ee_visual("left")

    @property
    def default_controller_config(self):
        return {
            "right": "default_rby1",
            "left": "default_rby1",
            "head": "default_rby1_head",
            "torso": "default_rby1_torso",
        }

    @property
    def init_qpos(self):
        # 22 joints (6 torso + 7 right arm + 7 left arm + 2 head)
        init_qpos = np.zeros(22)
        init_qpos[6:13] = _RIGHT_ARM_QPOS
        init_qpos[13:20] = _LEFT_ARM_QPOS
        return init_qpos

    @property
    def default_base(self):
        return "NoActuationBase"


class RBY1BothArms(RBY1):
    """
    RBY1 with only both arms active. Wheels, torso, and head are fixed.
    Uses OSC_POSE controller for Cartesian delta control.
    """

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("wheel")
        self._remove_joint_actuation("head")
        self._remove_joint_actuation("torso")
        self._remove_free_joint()
        # Both arms get grippers → remove static visuals on both sides
        self._remove_static_ee_visual("right")
        self._remove_static_ee_visual("left")

    @property
    def default_controller_config(self):
        return {
            "right": "default_rby1_single",
            "left": "default_rby1_single",
        }

    @property
    def init_qpos(self):
        # 14 joints (7 right arm + 7 left arm)
        init_qpos = np.zeros(14)
        init_qpos[0:7] = _RIGHT_ARM_QPOS
        init_qpos[7:14] = _LEFT_ARM_QPOS
        return init_qpos


# Keep old name as alias
RBY1ArmsOnly = RBY1BothArms


class RBY1RightArm(RBY1):
    """
    RBY1 with only the right arm active. Single-arm variant.
    Uses OSC_POSE controller for Cartesian delta control (from LIBERO).
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("wheel")
        self._remove_joint_actuation("head")
        self._remove_joint_actuation("torso")
        self._remove_joint_actuation("left_arm")
        self._remove_free_joint()
        # Right arm gets gripper → remove right static visuals (avoid double)
        self._remove_static_ee_visual("right")
        # Left arm has no gripper → keep left static visuals

    @property
    def default_gripper(self):
        return {"right": "RBY1Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_rby1_single"}

    @property
    def init_qpos(self):
        return _RIGHT_ARM_QPOS.copy()

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        return {"right": "right_eef"}


class RBY1LeftArm(RBY1):
    """
    RBY1 with only the left arm active. Single-arm variant.
    Uses OSC_POSE controller for Cartesian delta control (from LIBERO).

    Note: robosuite's single-arm infrastructure expects arm key "right",
    so the physical left arm is mapped to the "right" slot internally.
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self._remove_joint_actuation("wheel")
        self._remove_joint_actuation("head")
        self._remove_joint_actuation("torso")
        self._remove_joint_actuation("right_arm")
        self._remove_free_joint()
        # Left arm gets gripper → remove left static visuals (avoid double)
        self._remove_static_ee_visual("left")
        # Right arm has no gripper → keep right static visuals

    @property
    def default_gripper(self):
        return {"right": "RBY1Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_rby1_single"}

    @property
    def init_qpos(self):
        return _LEFT_ARM_QPOS.copy()

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        return {"right": "left_eef"}


# ── Backward-compatibility aliases ──
RBY1Single = RBY1RightArm  # Used by original rby1_LIBERO code
