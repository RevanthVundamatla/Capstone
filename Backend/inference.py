"""
DDPG Underwater Image Restoration - Inference

Runs the trained DDPG model on one underwater image.

Example:
    python inference.py --input data/UIEB/raw/9_img_.png

With reference image:
    python inference.py --input data/UIEB/raw/9_img_.png --reference data/UIEB/reference/9_img_.png
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import torch

from ddpg.agent import DDPGAgent
from enhancement.environment import UnderwaterEnhanceEnv
from metrics.metrics import all_metrics


# ============================================================
# HELPERS
# ============================================================

def _to_python(obj):
    """
    Recursively convert NumPy / torch values to plain Python types
    so the result is JSON serializable (needed by Flask's jsonify).
    """

    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    return obj


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path: str) -> np.ndarray:
    """
    Load image as RGB float32 in [0, 1].
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    return image


# ============================================================
# IMAGE SAVING
# ============================================================

def save_image(path: str, image: np.ndarray):
    """
    Save RGB float32 image in [0, 1].
    """

    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success = cv2.imwrite(path, image)

    if not success:
        raise IOError(f"Could not save image: {path}")


# ============================================================
# DDPG RESTORATION
# ============================================================

def restore_image(
    agent: DDPGAgent,
    image: np.ndarray,
    max_steps: int = 3,
    reference: np.ndarray | None = None,
):
    """
    Run the trained DDPG agent.

    image:
        HxWx3 float32 image in [0, 1]

    reference:
        Optional ground-truth reference image.

    Returns:
        restored_image
        metrics
        actions_taken
    """

    t_total = time.time()

    env = UnderwaterEnhanceEnv(max_steps=max_steps)

    state = env.reset(image, reference)

    actions_taken = []

    t_loop = time.time()

    for step in range(max_steps):

        t_step = time.time()

        with torch.no_grad():
            action = agent.select_action(state, noise_std=0.0)

        state, reward, done, info = env.step(action)

        actions_taken.append({
            "step": step + 1,
            "action": _to_python(action),
            "reward": float(reward),
            "metrics": _to_python(info),
        })

        print(
            f"[inference] step {step + 1}/{max_steps} "
            f"took {time.time() - t_step:.2f}s",
            flush=True,
        )

        if done:
            break

    print(
        f"[inference] environment loop took {time.time() - t_loop:.2f}s",
        flush=True,
    )

    restored = env.current.copy()

    t_metrics = time.time()

    metrics = all_metrics(restored, reference)

    print(
        f"[inference] all_metrics took {time.time() - t_metrics:.2f}s",
        flush=True,
    )

    print(
        f"[inference] restore_image total {time.time() - t_total:.2f}s",
        flush=True,
    )

    return (
        restored,
        metrics,
        actions_taken,
    )


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="DDPG Underwater Image Restoration"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to degraded underwater image",
    )

    parser.add_argument(
        "--reference",
        default=None,
        help="Optional path to reference image",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_ddpg_actor_critic.pt",
        help="Path to trained DDPG checkpoint",
    )

    parser.add_argument(
        "--output",
        default="outputs/restored.jpg",
        help="Path for restored output image",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of DDPG enhancement passes",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    print()
    print("=" * 70)
    print("        DDPG UNDERWATER IMAGE RESTORATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate checkpoint
    # --------------------------------------------------------

    print()
    print("[1/5] Loading trained model...")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{args.checkpoint}"
        )

    print(f"Checkpoint: {args.checkpoint}")

    agent = DDPGAgent(device=args.device)
    agent.load(args.checkpoint)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load input image
    # --------------------------------------------------------

    print()
    print("[2/5] Loading input image...")
    print(f"Input: {args.input}")

    image = load_image(args.input)

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    # --------------------------------------------------------
    # Load reference image
    # --------------------------------------------------------

    reference = None

    if args.reference:

        print()
        print("[3/5] Loading reference image...")
        print(f"Reference: {args.reference}")

        reference = load_image(args.reference)

        if image.shape != reference.shape:
            raise ValueError(
                "Input and reference image dimensions do not match.\n"
                f"Input: {image.shape}\n"
                f"Reference: {reference.shape}"
            )

        print("Reference loaded successfully.")

    else:

        print()
        print("[3/5] No reference image supplied.")
        print("Only no-reference metrics will be calculated.")

    # --------------------------------------------------------
    # Run DDPG
    # --------------------------------------------------------

    print()
    print("[4/5] Running DDPG restoration...")
    print(f"Enhancement passes: {args.steps}")

    restored, metrics, actions = restore_image(
        agent=agent,
        image=image,
        max_steps=args.steps,
        reference=reference,
    )

    print("Restoration completed.")

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    save_image(args.output, restored)

    print()
    print(f"Restored image saved to:\n{args.output}")

    # --------------------------------------------------------
    # Display actions
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("DDPG ENHANCEMENT ACTIONS")
    print("-" * 70)

    for item in actions:

        print(f"\nStep {item['step']}")
        print(f"Reward: {item['reward']:.6f}")
        print("Action:")
        print(np.round(item["action"], 4))

    # --------------------------------------------------------
    # Display metrics
    # --------------------------------------------------------

    print()
    print("[5/5] Evaluation")
    print()
    print("-" * 70)
    print("RESTORATION METRICS")
    print("-" * 70)

    if metrics is None:

        print("No metrics available.")

    else:

        for name, value in metrics.items():

            if value is None:
                continue

            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):

                if name.lower() == "psnr":
                    print(f"{name.upper():<8}: {float(value):.4f} dB")
                else:
                    print(f"{name.upper():<8}: {float(value):.4f}")

            else:

                print(f"{name.upper():<8}: {value}")

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("              INFERENCE COMPLETE")
    print("=" * 70)
    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
