"""Utils for evaluating policies in LIBERO simulation environments ported from OpenVLA."""

import math
import os
import cv2
import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from robot_utils import (
    DATE,
    DATE_TIME,
)


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_names": ["agentview", "robot0_eye_in_head", "robot0_eye_in_right_hand"]}
    env = OffScreenRenderEnv(**env_args)
    print(env.env)       # OffScreenRenderEnv 내부 env
    print(env.seed)
    #try:
    #    env.reset(seed=0)   # 새로운 방식
    #except TypeError:
    #    if hasattr(env, "seed") and callable(env.seed):
    #        env.seed(0)    # 구버전 fallback   
    #env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def resize_image(img, resize_size):
    """
    Takes a numpy array corresponding to a single image and returns the resized image as a numpy array.

    Args:
        img (np.ndarray): Input image, assumed dtype=np.uint8
        resize_size (tuple): (height, width) of desired output size

    Returns:
        np.ndarray: Resized image, dtype=np.uint8
    """
    assert isinstance(resize_size, tuple) and len(resize_size) == 2

    # OpenCV expects size as (width, height)
    resized_img = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LANCZOS4)

    # Clip and cast to uint8 to mimic tf behavior
    resized_img = np.clip(np.round(resized_img), 0, 255).astype(np.uint8)

    return resized_img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_head_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    #img = np.rot90(img, -1)
    img = resize_image(img, resize_size)
    return img

def get_libero_wrist_image(obs, resize_size):
    """Extracts wrist camera image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["robot0_eye_in_right_hand_image"]
    #img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.rot90(img, 3)
    img = resize_image(img, resize_size)
    return img

# ✅ [새 함수 추가] 녹화용 전체 시점 함수
def get_libero_agentview_image(obs, resize_size):
    """녹화용으로 전체 시점(AgentView)을 가져오는 함수"""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    
    # 전체를 조망하는 AgentView 사용
    img = obs["agentview_image"]
    
    img = img[::-1, ::-1] 
    
    img = resize_image(img, resize_size)
    return img
    
def save_rollout_video(rollout_images, idx, success, task_description, checkpoint, task, trajectory=None, color=(255,0,255)):
    """Saves an MP4 replay of an episode, optionally overlaying a trajectory (list of (x, y) pixel tuples) on each frame."""
    rollout_dir = f"./rollouts/{DATE}/{task}/{checkpoint}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    # Overlay trajectory if provided (purple)
    for i, img in enumerate(rollout_images):
        img_overlay = img.copy()
        if trajectory is not None and len(trajectory) > 1:
            pts = np.array(trajectory[:i+1], dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(img_overlay, [pts], isClosed=False, color=(255,0,255), thickness=2)
            if len(pts) > 0:
                cv2.circle(img_overlay, tuple(pts[-1]), 4, (255,0,255), -1)
        video_writer.append_data(img_overlay)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
