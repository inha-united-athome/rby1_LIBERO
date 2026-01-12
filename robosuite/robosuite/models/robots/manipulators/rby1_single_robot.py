import numpy as np
from robosuite.models.robots.manipulators.rby1_manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion
from scipy.spatial.transform import Rotation as R

class RBY1Single(ManipulatorModel):
    """
    RBY1 Single Arm Robot (Right Arm Only)
    """
    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/rby1/robot.xml"), idn=idn)
        self._remove_joint_actuation("head")
        self._remove_joint_actuation("torso")
        self._remove_joint_actuation("wheel")
        self._remove_joint_actuation("left")
        self._remove_free_joint()

    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        return {"right": "RBY1Gripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_rby1_single"}

    @property
    def init_qpos(self):
        """
        초기 관절 값 설정 - Panda와 유사한 EEF 위치/자세
        
        Panda EEF: Position≈[-0.1, 0.0, 1.0], Z-axis pointing down (-0.99)
        
        조인트 리미트:
          arm_0: [-2.356, 2.356]  - 어깨 yaw (-)일수록 위로 치켜듬
          arm_1: [-3.142, 0.050]  - 어깨 pitch (+) 일수록 몸쪽에 가까움
          arm_2: [-2.094, 2.094]  - 어깨 roll  
          arm_3: [-2.618, 0.010]  - 팔꿈치 - 일수록 접힘
          arm_4: [-6.283, 6.283]  - 손목 roll
          arm_5: [-1.745, 2.007]  - 손목 pitch (그리퍼 방향 결정!) (+) 일수록 안쪽으로 접힘
          arm_6: [-2.967, 2.967]  - 손목 yaw
        """
        # 그리퍼가 아래를 향하고 테이블 중앙 위에 위치
        # Panda EEF 목표: Position≈[-0.2, 0.0, 1.0], Euler≈[0, 7, 0]
        # 
        # arm_0: 어깨 yaw - Y축 위치 조정
        # arm_1: 어깨 pitch - 위/아래
        # arm_2: 어깨 roll - 팔 회전
        # arm_3: 팔꿈치 - 팔 접힘
        # arm_4: 손목 roll
        # arm_5: 손목 pitch - 그리퍼 위/아래 방향
        # arm_6: 손목 yaw
        #
        # 최적화 결과 (Euler ≈ [0.1, 6.5, 3.1], Panda의 [0, 7, 0]과 유사):
        # Position: [-0.36, -0.17, 1.05], Z-axis: [0.11, 0.01, 0.99]
        r_arm_qpos = np.array([-0.35, -0.60, -1.10, -1.00, -3.00, -1.74, -2.00])
        return r_arm_qpos

    @property
    def base_xpos_offset(self):
        """
        로봇 베이스 위치 오프셋
        
        - Panda: -0.16 - table_length/2 = -0.56 (table_length=0.8)
        - RBY1: -0.2 - table_length/2 = -0.60
        
        RBY1 바퀴는 테이블 다리 옆(X≈-0.37)에 위치하며, 
        테이블 상판(Z>0.75)과는 충돌하지 않음
        """
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
        return "single"

    @property
    def _eef_name(self):
        return {"right": "right_hand_eefd"}

    @property
    def gripper_mount_quat_offset(self):
        return {}
