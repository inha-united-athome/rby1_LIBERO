"""
RBY1 v1.1 Robot in robosuite - Interactive Visualization Demo

Merged implementation:
  - Our multi-variant support + transparent collision geoms + visual gripper meshes
  - LIBERO's OSC_POSE controller + optimized init_qpos (gripper pointing down)
  - LIBERO table type offsets for LIBERO benchmark compatibility

Available robot variants:
  - "RBY1RightArm"  : Right arm only, single-arm (7 DOF, OSC_POSE)
  - "RBY1LeftArm"   : Left arm only, single-arm (7 DOF, OSC_POSE)
  - "RBY1BothArms"  : Both arms, bimanual (14 DOF, OSC_POSE)

Usage:
  python test_rby1_robosuite.py                             # default: RBY1RightArm
  python test_rby1_robosuite.py --variant RBY1LeftArm       # left arm only
  python test_rby1_robosuite.py --variant RBY1BothArms      # both arms
  python test_rby1_robosuite.py --task Stack                # different task (bimanual only)
  python test_rby1_robosuite.py --no-viewer                 # headless (no GUI)
"""
import argparse
import os
import time

import numpy as np

# Ensure DISPLAY is set for X11
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":1"

import robosuite


def main():
    parser = argparse.ArgumentParser(description="RBY1 robosuite interactive demo")
    parser.add_argument("--variant", default="RBY1RightArm",
                        choices=["RBY1RightArm", "RBY1LeftArm", "RBY1BothArms"],
                        help="Robot variant to use")
    parser.add_argument("--task", default="Lift",
                        choices=["Lift", "Stack", "NutAssembly", "NutAssemblySquare",
                                 "NutAssemblyRound", "PickPlace", "Door", "Wipe"],
                        help="Task environment")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps to run")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless without GUI")
    args = parser.parse_args()

    use_viewer = not args.no_viewer

    print(f"=== RBY1 robosuite Demo ===")
    print(f"Robot : {args.variant}")
    print(f"Task  : {args.task}")
    print(f"Viewer: {'MuJoCo native' if use_viewer else 'headless'}")

    # Create environment (always headless from robosuite's perspective;
    # we use MuJoCo's native viewer for visualization instead)
    env = robosuite.make(
        args.task,
        robots=args.variant,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )

    action_dim = env.action_spec[0].shape[0]
    print(f"\nAction dim: {action_dim}")

    obs = env.reset()
    print(f"Obs keys  : {sorted(obs.keys())}")

    # --- Launch MuJoCo native interactive viewer ---
    viewer = None
    if use_viewer:
        import mujoco
        import mujoco.viewer

        # Get the underlying MuJoCo model and data from robosuite
        model = env.sim.model._model
        data = env.sim.data._data

        viewer = mujoco.viewer.launch_passive(model, data)
        # Set a nice default camera angle
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -25
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        print("\n✅ MuJoCo viewer opened! Drag to rotate, scroll to zoom, Ctrl+drag to pan.")
        print("   Close the viewer window to stop.\n")

    print(f"Running {args.steps} steps...")
    total_reward = 0.0
    t0 = time.time()

    for i in range(args.steps):
        # Check if viewer was closed
        if viewer is not None and not viewer.is_running():
            print(f"\nViewer closed at step {i}.")
            break

        # Small random actions to see the robot move
        action = np.random.uniform(-0.05, 0.05, size=action_dim)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Sync viewer with simulation state
        if viewer is not None:
            viewer.sync()

        if done:
            print(f"  Episode done at step {i+1}, resetting...")
            obs = env.reset()

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            print(f"  Step {i+1}/{args.steps}  reward={total_reward:.4f}  fps={fps:.1f}")

    elapsed = time.time() - t0
    print(f"\nDone! {i+1} steps in {elapsed:.1f}s ({(i+1)/elapsed:.1f} fps), total reward: {total_reward:.4f}")

    if viewer is not None:
        viewer.close()
    env.close()


if __name__ == "__main__":
    main()
