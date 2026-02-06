import argparse
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import torch
from scipy.spatial.transform import Rotation as R
from rby1_libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    get_libero_agentview_image,
    save_rollout_video,
)
from robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)
import tqdm
import ast
from libero.libero import benchmark
from transformers import AutoProcessor, AutoModelForImageTextToText 
import math
import random


def crop_and_resize_pil(img: Image.Image, crop_scale: float) -> Image.Image:
    """
    Center‐crop a PIL image to crop_scale of its area,
    then resize back to the ORIGINAL image size.
    """
    w, h = img.size
    # sqrt(crop_scale) to get relative side length
    rel = math.sqrt(crop_scale)
    cw, ch = int(w * rel), int(h * rel)
    left = (w - cw) // 2
    top  = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    # resize back to the original dimensions (w, h)
    return cropped.resize((w, h), Image.BILINEAR)


def center_crop_image(img: Image.Image) -> Image.Image:
    # fixed 0.9 area scale
    return crop_and_resize_pil(img, 0.9)


def renormalize_action_for_controller(action, unnorm_key):
    """
    Re-normalize MolmoAct action from unnormalized space back to [-1, 1] for the controller.
    
    MolmoAct outputs actions that are unnormalized using q01/q99 statistics.
    But the OSC controller expects actions in [-1, 1] range, which it then scales
    using output_max/output_min.
    
    This function converts the unnormalized action back to [-1, 1] range.
    
    Formula (inverse of unnormalization):
    normalized = 2 * (unnorm - q01) / (q99 - q01) - 1
    """
    if isinstance(action, list):
        action = np.array(action, dtype=float)
    
    # LIBERO spatial q01/q99 values (from model config)
    # These are the same values used in parse_action for unnormalization
    q01_q99_stats = {
        "libero_spatial_no_noops_modified": {
            "q01": np.array([-0.7454732114076613, -0.6616071462631226, -0.9375, 
                           -0.1071428582072258, -0.20678570866584778, -0.1842857152223587, 0.0]),
            "q99": np.array([0.9375, 0.8758928775787354, 0.9321428537368774,
                           0.1039285734295845, 0.17678570747375488, 0.14571428298950195, 1.0])
        },
        "libero_object_no_noops_modified": {
            "q01": np.array([-0.8125, -0.5625, -0.9375, -0.125, -0.1875, -0.1875, 0.0]),
            "q99": np.array([0.9375, 0.875, 0.9375, 0.125, 0.1875, 0.1875, 1.0])
        },
        "libero_goal_no_noops_modified": {
            "q01": np.array([-0.8125, -0.625, -0.9375, -0.125, -0.1875, -0.1875, 0.0]),
            "q99": np.array([0.9375, 0.875, 0.9375, 0.125, 0.1875, 0.1875, 1.0])
        },
        "libero_10_no_noops_modified": {
            "q01": np.array([-0.875, -0.625, -0.9375, -0.125, -0.1875, -0.1875, 0.0]),
            "q99": np.array([0.9375, 0.875, 0.9375, 0.125, 0.1875, 0.1875, 1.0])
        }
    }
    
    if unnorm_key not in q01_q99_stats:
        print(f"Warning: {unnorm_key} not in stats, using libero_spatial")
        unnorm_key = "libero_spatial_no_noops_modified"
    
    q01 = q01_q99_stats[unnorm_key]["q01"]
    q99 = q01_q99_stats[unnorm_key]["q99"]
    
    # Re-normalize position/orientation (indices 0-5), leave gripper (index 6) alone
    action_normalized = action.copy()
    for i in range(6):  # Only position and orientation, not gripper
        action_normalized[i] = 2.0 * (action[i] - q01[i]) / (q99[i] - q01[i]) - 1.0
    
    # Clip to [-1, 1] to ensure controller compatibility
    action_normalized[:6] = np.clip(action_normalized[:6], -1.0, 1.0)
    
    return action_normalized


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Convert to numpy array if it's a list
    if isinstance(action, list):
        action = np.array(action)
    
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    # Convert to numpy array if it's a list
    if isinstance(action, list):
        action = np.array(action)
    
    action[..., -1] = action[..., -1] * -1.0
    return action


def transform_orientation_action_for_rby1(action):
    """
    Transform delta action from Panda EEF frame to RBY1 EEF frame.
    
    NOTE (2026-01-08): After removing gripper_mount_quat_offset from the robot model,
    orientation transformation is NO LONGER needed. The RBY1 EEF now has a similar
    orientation to Panda (not exactly the same, but close enough for delta control).
    
    Keeping this function as identity transform for now.
    If issues persist, consider re-enabling the rotation transformation.
    
    Original problem:
    - Panda EEF Z-axis points DOWN (gripper opens downward, like claw machine)
    - RBY1 EEF Z-axis points UP/sideways
    - MolmoAct was trained on Panda, so delta actions are in Panda's perspective
    """
    if isinstance(action, list):
        action = np.array(action, dtype=float)
    
    # DISABLED: No transformation needed after removing gripper_mount_quat_offset
    # The rotation matrix that was previously used:
    # R_panda_to_rby1 = np.array([
    #     [1,  0,  0],
    #     [0, -1,  0],
    #     [0,  0, -1]
    # ])
    # delta_ori = action[3:6].copy()
    # action[3:6] = R_panda_to_rby1 @ delta_ori
    
    return action


def initialize_rby1_orientation(env, num_init_steps=50):
    """
    Initialize RBY1 - simplified version without orientation adjustment.
    
    Since adjusting joint values doesn't easily achieve Panda-like orientation,
    we'll focus on just ensuring stable initialization with gripper open.
    
    The orientation difference will be handled by the action transformation.
    
    Args:
        env: The robosuite environment
        num_init_steps: Number of warmup steps for stabilization
        
    Returns:
        obs: Final observation after initialization
    """
    from scipy.spatial.transform import Rotation as R
    
    print("Initializing RBY1...")
    
    robot = env.env.robots[0]
    sim = robot.sim
    
    # Print initial state for debugging
    try:
        initial_eef_pose = robot.recent_ee_pose["right"].last
        initial_eef_pos = initial_eef_pose[:3]
        initial_eef_ori = initial_eef_pose[3:]  # quaternion wxyz
        print(f"Initial EEF position: {initial_eef_pos.round(4)}")
        
        # Convert to euler for readability
        quat_xyzw = np.array([initial_eef_ori[1], initial_eef_ori[2], initial_eef_ori[3], initial_eef_ori[0]])
        euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
        print(f"Initial EEF euler (xyz degrees): {euler.round(2)}")
        
        # Calculate gripper Z-axis direction
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        gripper_z = rot_mat @ np.array([0, 0, 1])
        print(f"Gripper Z-axis direction: {gripper_z.round(3)} (Panda: [0, 0, -1])")
    except Exception as e:
        print(f"Cannot get initial EEF pose: {e}")
    
    # Check gripper state
    try:
        gripper_idx = robot._ref_gripper_joint_pos_indexes
        if isinstance(gripper_idx, dict):
            gripper_idx = gripper_idx.get("right", list(gripper_idx.values())[0])
        gripper_qpos = sim.data.qpos[gripper_idx]
        print(f"Initial gripper joint positions: {gripper_qpos}")
    except Exception as e:
        print(f"Cannot get gripper qpos: {e}")
    
    obs = None
    # Warmup steps to stabilize and ensure gripper is open
    for i in range(num_init_steps):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # no movement, gripper open
        obs, _, done, _ = env.step(action)
        if done:
            break
    
    # Print final state
    try:
        final_eef_pose = robot.recent_ee_pose["right"].last
        final_eef_pos = final_eef_pose[:3]
        final_eef_ori = final_eef_pose[3:]
        print(f"Final EEF position: {final_eef_pos.round(4)}")
        
        quat_xyzw = np.array([final_eef_ori[1], final_eef_ori[2], final_eef_ori[3], final_eef_ori[0]])
        euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
        print(f"Final EEF euler (xyz degrees): {euler.round(2)}")
        
        # Check gripper state
        gripper_idx = robot._ref_gripper_joint_pos_indexes
        if isinstance(gripper_idx, dict):
            gripper_idx = gripper_idx.get("right", list(gripper_idx.values())[0])
        gripper_qpos = sim.data.qpos[gripper_idx]
        print(f"Final gripper joint positions: {gripper_qpos}")
    except Exception as e:
        print(f"Cannot get final state: {e}")
    
    print("Initialization complete")
    return obs

def apply_chat_template(processor: AutoProcessor, text: str):
    messages = [
        {
            "role": "user",
            "content": [dict(type="text", text=text)]
        }
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return prompt

def scale_pt(self, pt, w, h):
    """
    Convert a point whose coordinates are in 0–255 space
    to image-pixel space (0‥w-1, 0‥h-1).
    """
    x, y = pt
    return (int(round(x / 255.0 * (w - 1))),
            int(round(y / 255.0 * (h - 1))))
    

def step(img, wrist_img, language_instruction, model, processor, unnorm_key):
    """
    Run the multimodal model to get a text, parse out the 8×7 action matrix,
    unnormalize, then temporally aggregate the first 6 DOFs (dims 0–5) while using
    the latest value for DOF 6. Return a single aggregated 7-D action vector and
    the annotated image.
    """
         
    image = Image.fromarray(img)
    wrist = Image.fromarray(wrist_img)
    image = center_crop_image(image)
    wrist = center_crop_image(wrist)
    imgs = [image, wrist]


    prompt = (
        f"The task is {language_instruction}. "
        "What is the action that the robot should take. "
        f"To figure out the action that the robot should take to {language_instruction}, "
        "let's think through it step by step. "
        "First, what is the depth map for the first image? "
        "Second, what is the trajectory of the end effector in the first image? "
        "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
        "along with other images from different camera views as additional information, "
        "what is the action that the robot should take?"
    )
    
        
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [dict(type="text", text=prompt)]
            }
        ], 
        tokenize=False, 
        add_generation_prompt=True,
    )
        
    inputs = processor(
        images=[imgs],
        text=text,
        padding=True,
        return_tensors="pt",
    )


    inputs = {k: v.to(model.device) for k, v in inputs.items()}


    # generate output
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=512)

    # only get generated tokens; decode them to text
    generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
    generated_text = processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # print the generated text
    print(f"generated text: {generated_text}")


    depth = model.parse_depth(generated_text)
    print(f"generated depth perception tokens: {depth}")
    
    trace = model.parse_trace(generated_text)
    print(f"generated visual reasoning trace: {trace}")


    # Try task-specific unnorm_key first; fall back to whatever the model has
    try:
        action = model.parse_action(generated_text, unnorm_key=unnorm_key)
    except ValueError:
        available_keys = list(model.norm_stats.keys()) if hasattr(model, 'norm_stats') else []
        print(f"Warning: unnorm_key={unnorm_key!r} not available. Available: {available_keys}")
        if available_keys:
            fallback_key = available_keys[0]
            print(f"Falling back to unnorm_key={fallback_key!r}")
            action = model.parse_action(generated_text, unnorm_key=fallback_key)
        else:
            raise
    print(f"generated action: {action}")

    if (
        action is None
        or (isinstance(action, (list, tuple)) and len(action) == 0)
        or (isinstance(action, np.ndarray) and action.size == 0)
    ):
        raise ValueError("parse_action produced no action (None/empty).")
    annotated = np.array(img.copy())



    return action, annotated, trace



# @draccus.wrap()
def eval_libero(args, processor, model, task_suite_name, checkpoint, seed, model_family, num_trials_per_task, num_steps_wait) -> None:

    set_seed_everywhere(seed)



    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    print(f"Task suite: {task_suite_name}")


    # Get expected image dimensions
    resize_size = get_image_resize_size()


    # Start evaluation
    total_episodes, total_successes = 0, 0
    for _ in tqdm.tqdm(range(1)):
        # Get task
        task_id = args.task_id
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(num_trials_per_task)):
            last_gripper_state = -1
       
       
         
            print(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            # === RBY1: Initialize gripper orientation to point downward ===
            # Panda starts with gripper pointing down, but RBY1 points sideways
            # Apply initialization actions to rotate gripper to match Panda's pose
            # Note: This is RBY1-specific evaluation script, so always apply
            obs = initialize_rby1_orientation(env, num_init_steps=30)

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            replay_combined = []
            if task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
                unnorm_key = "libero_spatial_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
                unnorm_key = "libero_object_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
                unnorm_key = "libero_goal_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
                unnorm_key = "libero_10_no_noops_modified"
                print(f"Max steps: {max_steps}")
            elif task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
                print(f"Max steps: {max_steps}")

            print(f"Starting episode {task_episodes+1}...")
   
            
            timestep = 0
            outer_done = False
         

            # === Trajectory overlay용 픽셀 trajectory 저장 ===
            pixel_trajectory = []
            while t < max_steps + num_steps_wait and not outer_done:
                # 1) Warm-up: ignore its 'done'
                if t < num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(model_family))
                    t += 1
                    continue

                # 2) step action
                img = get_libero_image(obs, resize_size)
                wrist_img = get_libero_wrist_image(obs, resize_size)
                agent_img = get_libero_agentview_image(obs, resize_size)
                wait = False
                traj = None
                try:
                    action_matrix, annotated_image, traj = step(agent_img, wrist_img, task_description, model, processor, unnorm_key)
                except Exception as e:
                    print(e)
                    action_matrix = np.zeros((1, 7), dtype=float)
                    action_matrix[:, -1] = last_gripper_state
                    annotated_image = agent_img
                    wait = True
                    print(f"error: {e}")

                if annotated_image is None:
                    annotated_image = agent_img

                # === 현재 스텝의 trajectory 파싱 ===
                current_traj_points = None
                if traj is not None and isinstance(traj, list) and len(traj) > 0:
                    # traj는 [[[x1,y1], [x2,y2], ...]] 형태이므로 첫 번째를 사용
                    if isinstance(traj[0], list) and len(traj[0]) > 0:
                        if isinstance(traj[0][0], list):
                            # traj = [[[x,y], [x,y], ...]]
                            current_traj_points = traj[0]
                        else:
                            # traj = [[x,y], [x,y], ...]
                            current_traj_points = traj
                
                # === annotated_image에 현재 스텝의 trajectory만 시각화 ===
                annotated_image = np.array(annotated_image.copy())
                if current_traj_points is not None and len(current_traj_points) > 1:
                    for i in range(len(current_traj_points) - 1):
                        p1 = tuple(map(int, current_traj_points[i]))
                        p2 = tuple(map(int, current_traj_points[i + 1]))
                        cv2.line(annotated_image, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                    # 시작점과 끝점 표시
                    cv2.circle(annotated_image, tuple(map(int, current_traj_points[0])), 5, (0, 255, 0), -1)  # 시작점 (녹색)
                    cv2.circle(annotated_image, tuple(map(int, current_traj_points[-1])), 5, (255, 0, 0), -1)  # 끝점 (빨간색)
                    print(f"Trajectory points: {len(current_traj_points)}")

                # combine head + wrist side-by-side
                intention_frame = np.hstack([annotated_image, wrist_img, img])
                replay_combined.append(intention_frame)


                action_num = 0
                # 3) Execute each of the N actions until done
                for single_action in action_matrix:
                    
                    if isinstance(single_action, str):
                        single_action = ast.literal_eval(single_action)
                    
                    single_action = np.asarray(single_action)
                    
                    # === MolmoAct action 처리 (Panda와 동일하게) ===
                    # 재정규화 없이 unnormalized action 그대로 사용
                    # OSC 컨트롤러가 action * output_max로 스케일링
                    print(f"Raw action: {single_action[:6].round(4)}")
                    
                    # === RBY1 액션 변환 ===
                    # MolmoAct는 Panda 로봇용으로 학습되었으므로, RBY1에 맞게 변환 필요
                    single_action = np.array(single_action, dtype=float)
                    single_action = normalize_gripper_action(single_action, binarize=True)
                    single_action = invert_gripper_action(single_action)
                    print(f"After gripper norm: {single_action[:6].round(4)}, gripper: {single_action[-1]:.4f}")
                    
                    # === 그리퍼 액션 처리 ===
                    # LIBERO 데이터셋: 0=close, 1=open
                    # normalize_gripper_action: 0 → -1, 1 → +1
                    # invert_gripper_action: -1 → +1 (close), +1 → -1 (open)
                    # robosuite convention: -1=open, +1=close
                    # FORCE CLOSE GRIPPER for grasping
                    #single_action[-1] = 1.0  # Force close gripper
                    
                    print(f"Final action: {single_action.round(4)}, gripper: {single_action[-1]} ({'close' if single_action[-1] > 0 else 'open'})")
                    
                    # === DEBUG: Print EEF position before step ===
                    try:
                        eef_pos_before = env.env.robots[0].recent_ee_pose["right"].last[:3]
                        print(f"EEF pos before step: {eef_pos_before.round(4)}")
                    except Exception as e:
                        print(f"Cannot get EEF pos: {e}")
                    
                    obs, _, done, _ = env.step(single_action)
                    
                    # === DEBUG: Print EEF position after step ===
                    try:
                        eef_pos_after = env.env.robots[0].recent_ee_pose["right"].last[:3]
                        print(f"EEF pos after step: {eef_pos_after.round(4)}, delta: {(eef_pos_after - eef_pos_before).round(4)}")
                    except Exception as e:
                        print(f"Cannot get EEF pos: {e}")
                    
                    visualize = get_libero_agentview_image(obs, resize_size)
                    #visualize = get_libero_image(obs, resize_size)
                    visualize_wrist = get_libero_wrist_image(obs, resize_size)
                    agent_img = get_libero_image(obs, resize_size)

                    visualize_annotated = np.array(visualize.copy())
                    # 기존 프레임에 직접 trajectory를 그리지 않고, 원본 프레임만 저장
                    combined_frame = np.hstack([visualize, visualize_wrist, agent_img])
                    replay_combined.append(combined_frame)

                    action_num += 1
   
                    if done:
                        outer_done = True
                        break
                
                # 4) Advance your loop counters
                timestep += 1
                print(f"wait: {wait}")
                if wait:
                    action_num = 1
                    
                print(f"action num: {action_num}")
                t += action_num


                if done:
                    task_successes += 1
                    total_successes += 1
                    break


            task_episodes += 1
            total_episodes += 1
            

            # Save a replay video of the episode
            #save_rollout_video(
            #    replay_images, total_episodes, success=done, task_description=task_description, checkpoint=checkpoint, task=task_suite_name
            #)

            # Save a replay video of the episode

            # Trajectory overlay 전달
            save_rollout_video(
                replay_combined, total_episodes, success=done, task_description=task_description+"_combined", checkpoint=checkpoint, task=task_suite_name, trajectory=pixel_trajectory
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

    




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     type=str, required=True)
    p.add_argument("--task_id",  type=int, required=False, default=None, 
                   help="Specific task ID (0-9). If not provided, will run all task IDs 0-9 for the specified task type.")
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()
    task_suite_name = f"libero_{args.task}"
    ckpt       = args.checkpoint
    seed = random.randint(1, 1999)

    set_seed_everywhere(seed)
    


    processor = AutoProcessor.from_pretrained(
        ckpt,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
        padding_side="left",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        ckpt,
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    model_family = ckpt.replace("/", "-")
    num_trials_per_task = 50
    num_steps_wait = 10  
    
    if args.task_id is not None:
        print(f"Running single task ID: {args.task_id}")
        eval_libero(args, processor, model, task_suite_name, ckpt, seed, model_family, num_trials_per_task, num_steps_wait)
    else:
        # Run all task IDs 0-9 for the specified task type
        print(f"Running all task IDs 0-9 for task type: {args.task}")
        for task_id in range(10):
            print(f"\n{'='*50}")
            print(f"Running task ID: {task_id}")
            print(f"{'='*50}")
            args.task_id = task_id
            eval_libero(args, processor, model, task_suite_name, ckpt, seed, model_family, num_trials_per_task, num_steps_wait)

if __name__ == "__main__":
    main()
