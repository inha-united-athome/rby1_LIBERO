#!/usr/bin/env python3
"""
RBY1 robosuite 자동 설치 스크립트.

robosuite 설치 경로에 RBY1 로봇/그리퍼 파일을 복사하고
등록 파일(__init__.py)을 자동으로 패치합니다.

Usage:
    python setup_robosuite.py              # 자동 감지
    python setup_robosuite.py --path /path/to/robosuite  # 경로 지정
"""

import argparse
import importlib
import os
import shutil
import sys
import re

# ── Files to copy ──
COPY_MAP = {
    # source (relative to this script) → destination (relative to robosuite root)
    "robosuite/robosuite/models/robots/manipulators/rby1_robot.py":
        "models/robots/manipulators/rby1_robot.py",
    "robosuite/robosuite/models/grippers/rby1_gripper.py":
        "models/grippers/rby1_gripper.py",
    "robosuite/robosuite/models/assets/grippers/rby1_gripper.xml":
        "models/assets/grippers/rby1_gripper.xml",
    "robosuite/robosuite/models/assets/robots/rby1/robot.xml":
        "models/assets/robots/rby1/robot.xml",
    "robosuite/robosuite/controllers/config/robots/default_rby1_single.json":
        "controllers/config/robots/default_rby1_single.json",
    "robosuite/robosuite/controllers/config/robots/default_rby1.json":
        "controllers/config/robots/default_rby1.json",
}

# Directories with .obj meshes to copy
MESH_DIRS = {
    "robosuite/robosuite/models/assets/grippers/meshes/rby1_gripper":
        "models/assets/grippers/meshes/rby1_gripper",
    "robosuite/robosuite/models/assets/robots/rby1/meshes":
        "models/assets/robots/rby1/meshes",
}

# ── Registration patches ──

# 1. models/robots/manipulators/__init__.py
MANIP_IMPORT = (
    "from .rby1_robot import RBY1, RBY1FixedLowerBody, RBY1ArmsOnly, "
    "RBY1BothArms, RBY1RightArm, RBY1LeftArm, RBY1Single"
)

# 2. models/grippers/__init__.py
GRIPPER_IMPORT = "from .rby1_gripper import RBY1Gripper, RBY1GripperBase"
GRIPPER_MAPPING_ENTRIES = {
    "RBY1Gripper": "RBY1Gripper",
    "RBY1GripperBase": "RBY1GripperBase",
}

# 3. robots/__init__.py  (ROBOT_CLASS_MAPPING)
ROBOT_CLASS_ENTRIES = {
    "RBY1":              "LeggedRobot",
    "RBY1FixedLowerBody":"LeggedRobot",
    "RBY1ArmsOnly":      "LeggedRobot",
    "RBY1BothArms":      "LeggedRobot",
    "RBY1RightArm":      "FixedBaseRobot",
    "RBY1LeftArm":       "FixedBaseRobot",
    "RBY1Single":        "FixedBaseRobot",
}

# 4. manipulator_model.py line ~36 patch
MANIP_MODEL_OLD = 'arms = self.__class__.arms'
MANIP_MODEL_NEW = 'arms = getattr(self.__class__, "arms", ["right"])'


def find_robosuite_path(user_path=None):
    """Find robosuite installation directory."""
    if user_path:
        if os.path.isdir(user_path) and os.path.exists(os.path.join(user_path, "models")):
            return user_path
        raise FileNotFoundError(f"Not a valid robosuite path: {user_path}")
    try:
        mod = importlib.import_module("robosuite")
        return os.path.dirname(mod.__file__)
    except ImportError:
        raise ImportError("robosuite not found. Install it first or use --path.")


def copy_files(repo_root, rs_root):
    """Copy RBY1 robot/gripper/mesh files into robosuite."""
    count = 0
    for src_rel, dst_rel in COPY_MAP.items():
        src = os.path.join(repo_root, src_rel)
        dst = os.path.join(rs_root, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        count += 1

    for src_dir_rel, dst_dir_rel in MESH_DIRS.items():
        src_dir = os.path.join(repo_root, src_dir_rel)
        dst_dir = os.path.join(rs_root, dst_dir_rel)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            if f.endswith(".obj"):
                shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
                count += 1

    print(f"  ✅ Copied {count} files")


def patch_file(filepath, check_str, patch_fn, desc):
    """Apply patch if not already applied."""
    with open(filepath, "r") as f:
        content = f.read()
    if check_str in content:
        print(f"  ⏭  Already patched: {desc}")
        return
    new_content = patch_fn(content)
    with open(filepath, "w") as f:
        f.write(new_content)
    print(f"  ✅ Patched: {desc}")


def patch_manipulators_init(rs_root):
    """Register RBY1 robot classes in manipulators/__init__.py."""
    filepath = os.path.join(rs_root, "models/robots/manipulators/__init__.py")

    def _patch(content):
        if "rby1_robot" not in content:
            content += f"\n{MANIP_IMPORT}\n"
        else:
            # Replace existing import line
            content = re.sub(
                r"from \.rby1.*?import.*?\n",
                MANIP_IMPORT + "\n",
                content,
            )
        return content

    patch_file(filepath, "RBY1RightArm", _patch, "manipulators/__init__.py")


def patch_grippers_init(rs_root):
    """Register RBY1Gripper in grippers/__init__.py."""
    filepath = os.path.join(rs_root, "models/grippers/__init__.py")

    def _patch(content):
        if "rby1_gripper" not in content:
            # Add import after last import
            lines = content.split("\n")
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("from .") or line.startswith("import "):
                    last_import_idx = i
            lines.insert(last_import_idx + 1, GRIPPER_IMPORT)
            content = "\n".join(lines)
        # Add to GRIPPER_MAPPING if it exists
        for name, var in GRIPPER_MAPPING_ENTRIES.items():
            entry = f'    "{name}": {var},'
            if entry not in content and "GRIPPER_MAPPING" in content:
                content = content.replace(
                    "GRIPPER_MAPPING = {",
                    f"GRIPPER_MAPPING = {{\n{entry}",
                )
        return content

    patch_file(filepath, "RBY1Gripper", _patch, "grippers/__init__.py")


def patch_robots_init(rs_root):
    """Register RBY1 variants in robots/__init__.py ROBOT_CLASS_MAPPING."""
    filepath = os.path.join(rs_root, "robots/__init__.py")

    def _patch(content):
        for name, cls in ROBOT_CLASS_ENTRIES.items():
            entry = f'    "{name}": {cls},'
            if entry not in content and "ROBOT_CLASS_MAPPING" in content:
                content = content.replace(
                    "ROBOT_CLASS_MAPPING = {",
                    f"ROBOT_CLASS_MAPPING = {{\n{entry}",
                )
        return content

    patch_file(filepath, '"RBY1":', _patch, "robots/__init__.py")


def patch_manipulator_model(rs_root):
    """Patch ManipulatorModel to support arms class attribute (left-arm fix)."""
    filepath = os.path.join(
        rs_root, "models/robots/manipulators/manipulator_model.py"
    )

    def _patch(content):
        return content.replace(MANIP_MODEL_OLD, MANIP_MODEL_NEW)

    patch_file(filepath, 'getattr(self.__class__, "arms"', _patch,
               "manipulator_model.py (left-arm support)")


def main():
    parser = argparse.ArgumentParser(description="Install RBY1 into robosuite")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to robosuite package directory")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    rs_root = find_robosuite_path(args.path)

    print(f"📦 robosuite path: {rs_root}")
    print(f"📂 repo root:      {repo_root}")
    print()

    print("[1/5] Copying RBY1 files...")
    copy_files(repo_root, rs_root)

    print("[2/5] Patching manipulators/__init__.py...")
    patch_manipulators_init(rs_root)

    print("[3/5] Patching grippers/__init__.py...")
    patch_grippers_init(rs_root)

    print("[4/5] Patching robots/__init__.py...")
    patch_robots_init(rs_root)

    print("[5/5] Patching manipulator_model.py...")
    patch_manipulator_model(rs_root)

    print()
    print("🎉 RBY1 설치 완료!")
    print()
    print("Available variants:")
    print("  - RBY1             : Full robot (wheels + torso + dual arms + head)")
    print("  - RBY1FixedLowerBody: No wheels, everything else active")
    print("  - RBY1BothArms     : Both arms only (OSC_POSE)")
    print("  - RBY1RightArm     : Right arm only (OSC_POSE)")
    print("  - RBY1LeftArm      : Left arm only (OSC_POSE)")
    print("  - RBY1Single       : Alias for RBY1RightArm (LIBERO compat)")
    print()
    print("Test with:")
    print('  python -c "import robosuite; env = robosuite.make(\\"Lift\\", robots=\\"RBY1RightArm\\"); env.reset(); print(\\"OK\\"); env.close()"')


if __name__ == "__main__":
    main()
