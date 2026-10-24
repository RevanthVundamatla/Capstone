
"""
DDPG training for underwater image restoration using UIEB.

Dataset:
    data/UIEB/raw
    data/UIEB/reference

Split:
    80% training
    10% validation
    10% testing

Metrics:
    PSNR, SSIM, UIQM, UCIQE, color cast

Features:
    - Reproducible dataset split
    - Raw-image baseline metrics
    - PSNR-prioritized checkpoint selection
    - Genuine validation history for frontend graphs
    - Separate best and latest checkpoints
    - Final evaluation on the held-out test split
"""

from __future__ import annotations

import argparse
import json
import os
import random

import cv2
import numpy as np

from ddpg.agent import DDPGAgent
from ddpg.replay_buffer import ReplayBuffer
from enhancement.environment import (
    UnderwaterEnhanceEnv,
    UIQM_NORM,
    UCIQE_NORM,
    color_cast,
)
from metrics.metrics import psnr, ssim, uiqm, uciqe


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
IMAGE_MAX_DIM = 256


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(path: str) -> np.ndarray:
    """Load an image as RGB float32 in [0, 1]."""

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    if IMAGE_MAX_DIM and max(h, w) > IMAGE_MAX_DIM:
        scale = IMAGE_MAX_DIM / max(h, w)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

    return img.astype(np.float32) / 255.0


def load_pair(raw_path: str, ref_path: str):
    """Load a raw/reference pair with matching dimensions."""

    raw_img = load_image(raw_path)
    ref_img = load_image(ref_path)

    if ref_img.shape != raw_img.shape:
        ref_img = cv2.resize(
            ref_img,
            (raw_img.shape[1], raw_img.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    return raw_img, ref_img


# =========================================================
# DATASET DIRECTORY RESOLUTION
# =========================================================

def resolve_image_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory does not exist: {path}")

    entries = os.listdir(path)

    if any(name.lower().endswith(IMG_EXTS) for name in entries):
        return path

    subdirs = [
        name
        for name in entries
        if os.path.isdir(os.path.join(path, name))
    ]

    if len(subdirs) == 1:
        return resolve_image_dir(os.path.join(path, subdirs[0]))

    return path


# =========================================================
# DATASET PAIRING
# =========================================================

def list_pairs(
    data_dir: str,
    raw_folder: str = "raw",
    ref_folder: str = "reference",
):
    raw_dir = resolve_image_dir(os.path.join(data_dir, raw_folder))
    ref_dir = resolve_image_dir(os.path.join(data_dir, ref_folder))

    raw_names = sorted(
        name for name in os.listdir(raw_dir)
        if name.lower().endswith(IMG_EXTS)
    )

    ref_names = sorted(
        name for name in os.listdir(ref_dir)
        if name.lower().endswith(IMG_EXTS)
    )

    if not raw_names or not ref_names:
        raise FileNotFoundError(
            f"No images found.\nRaw: {raw_dir}\nReference: {ref_dir}"
        )

    ref_by_lower_name = {
        name.lower(): name for name in ref_names
    }

    pairs = []

    for raw_name in raw_names:
        ref_name = ref_by_lower_name.get(raw_name.lower())

        if ref_name is not None:
            pairs.append((
                os.path.join(raw_dir, raw_name),
                os.path.join(ref_dir, ref_name),
            ))

    if not pairs:
        raise RuntimeError(
            "No matching filenames were found between raw and reference "
            "folders. Check the dataset pairing before training; "
            "automatic sorted pairing could pair unrelated images."
        )

    print(f"Paired {len(pairs)} images by filename.")
    return pairs


# =========================================================
# DATASET SPLIT
# =========================================================

def split_dataset(
    pairs,
    train_ratio=0.80,
    val_ratio=0.10,
    seed=42,
):
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return (
        pairs[:train_end],
        pairs[train_end:val_end],
        pairs[val_end:],
    )


# =========================================================
# METRIC HELPERS
# =========================================================

def calculate_metrics(image: np.ndarray, reference: np.ndarray) -> dict:
    return {
        "psnr": float(psnr(image, reference)),
        "ssim": float(ssim(image, reference)),
        "uiqm": float(uiqm(image)),
        "uciqe": float(uciqe(image)),
        "cast": float(color_cast(image)),
    }


def evaluate_agent(
    agent,
    pairs,
    max_steps=3,
    report_baseline=True,
):
    """
    Evaluate the agent and the original input on exactly the same pairs.
    All metrics are computed from actual images.
    """

    env = UnderwaterEnhanceEnv(max_steps=max_steps)

    restored_metrics = []
    baseline_metrics = []

    for index, (raw_path, ref_path) in enumerate(pairs, start=1):
        raw_img, ref_img = load_pair(raw_path, ref_path)

        baseline_metrics.append(
            calculate_metrics(raw_img, ref_img)
        )

        state = env.reset(raw_img.copy(), ref_img.copy())
        done = False

        while not done:
            action = agent.select_action(
                state,
                noise_std=0.0,
            )

            state, _, done, _ = env.step(action)

        restored = np.asarray(env.current, dtype=np.float32)
        restored = np.clip(restored, 0.0, 1.0)

        if restored.shape != ref_img.shape:
            restored = cv2.resize(
                restored,
                (ref_img.shape[1], ref_img.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        restored_metrics.append(
            calculate_metrics(restored, ref_img)
        )

        if index % 25 == 0:
            print(f"  Evaluated {index}/{len(pairs)} images")

    def mean_metrics(items):
        if not items:
            return {}

        return {
            key: float(np.mean([item[key] for item in items]))
            for key in items[0]
        }

    result = {
        "restored": mean_metrics(restored_metrics),
    }

    if report_baseline:
        result["baseline"] = mean_metrics(baseline_metrics)

    return result


# =========================================================
# CHECKPOINT SELECTION
# =========================================================

def validation_score(results: dict) -> float:
    """
    Prioritize PSNR, while retaining SSIM and perceptual metrics.
    This score is only for checkpoint selection.
    """

    psnr_score = np.clip(results["psnr"] / 30.0, 0.0, 1.0)
    ssim_score = np.clip(results["ssim"], 0.0, 1.0)
    uiqm_score = np.clip(results["uiqm"] / UIQM_NORM, 0.0, 1.0)
    uciqe_score = np.clip(results["uciqe"] / UCIQE_NORM, 0.0, 1.0)

    return float(
        0.70 * psnr_score
        + 0.15 * ssim_score
        + 0.10 * uiqm_score
        + 0.05 * uciqe_score
        - 0.10 * max(results["cast"], 0.0)
    )


def save_history(path: str, history: list):
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump({"metrics": history}, file, indent=2)


def print_metrics(title: str, metrics: dict):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"PSNR  : {metrics['psnr']:.4f} dB")
    print(f"SSIM  : {metrics['ssim']:.4f}")
    print(f"UIQM  : {metrics['uiqm']:.4f}")
    print(f"UCIQE : {metrics['uciqe']:.4f}")
    print(f"CAST  : {metrics['cast']:.4f}")


# =========================================================
# TRAINING
# =========================================================

def train(args):
    global IMAGE_MAX_DIM
    IMAGE_MAX_DIM = args.max_dim

    random.seed(args.seed)
    np.random.seed(args.seed)

    pairs = list_pairs(
        args.data_dir,
        args.raw_folder,
        args.ref_folder,
    )

    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs,
        train_ratio=0.80,
        val_ratio=0.10,
        seed=args.seed,
    )

    if not train_pairs or not val_pairs or not test_pairs:
        raise ValueError(
            "Training, validation, or test split is empty. "
            "Check the dataset size."
        )

    print("\nDataset split")
    print("------------------------------")
    print(f"Training       : {len(train_pairs)}")
    print(f"Validation     : {len(val_pairs)}")
    print(f"Test           : {len(test_pairs)}")
    print(f"Epochs         : {args.epochs}")
    print(f"Image max side : {IMAGE_MAX_DIM}")
    print(f"Device         : {args.device}")
    print("------------------------------")

    agent = DDPGAgent(
        device=args.device,
        gamma=args.gamma,
        tau=args.tau,
    )

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume}"
            )

        agent.load(args.resume)
        print(f"Resumed model from: {args.resume}")

    buffer = ReplayBuffer(capacity=args.buffer_size)
    env = UnderwaterEnhanceEnv(max_steps=args.max_steps)

    noise_std = args.noise_start
    best_val_score = -float("inf")
    history = []

    for path in (args.checkpoint, args.best_checkpoint, args.history_file):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    # Evaluate initial performance before any training.
    print("\nInitial validation performance")
    initial = evaluate_agent(
        agent,
        val_pairs,
        max_steps=args.max_steps,
    )
    print_metrics("Original input baseline", initial["baseline"])
    print_metrics("Initial agent output", initial["restored"])

    # Save initial metric record for the graph.
    initial_record = {
        "epoch": 0,
        "train_reward": 0.0,
        **initial["restored"],
        "baseline_psnr": initial["baseline"]["psnr"],
        "baseline_ssim": initial["baseline"]["ssim"],
        "validation_score": validation_score(initial["restored"]),
    }
    history.append(initial_record)
    save_history(args.history_file, history)

    # Select the best checkpoint using the initial validation result.
    best_val_score = initial_record["validation_score"]
    agent.save(args.best_checkpoint)
    print(f"Initial checkpoint saved: {args.best_checkpoint}")

    # Each epoch is one complete pass over the training split.
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_pairs)

        epoch_rewards = []

        for image_index, (raw_path, ref_path) in enumerate(
            train_pairs,
            start=1,
        ):
            raw_img, ref_img = load_pair(raw_path, ref_path)
            state = env.reset(raw_img, ref_img)

            episode_reward = 0.0

            for _ in range(args.max_steps):
                action = agent.select_action(
                    state,
                    noise_std=noise_std,
                )

                next_state, reward, done, _ = env.step(action)

                buffer.push(
                    state,
                    action,
                    reward,
                    next_state,
                    float(done),
                )

                state = next_state
                episode_reward += float(reward)

                if len(buffer) >= args.batch_size:
                    agent.train_step(
                        buffer,
                        batch_size=args.batch_size,
                    )

                if done:
                    break

            epoch_rewards.append(episode_reward)

            noise_std = max(
                args.noise_end,
                noise_std * args.noise_decay,
            )

        avg_reward = (
            float(np.mean(epoch_rewards))
            if epoch_rewards else 0.0
        )

        print(
            f"\nEpoch {epoch:4d}/{args.epochs} | "
            f"avg_reward={avg_reward:.4f} | "
            f"noise={noise_std:.3f} | "
            f"buffer={len(buffer)}"
        )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluated = evaluate_agent(
                agent,
                val_pairs,
                max_steps=args.max_steps,
            )

            metrics = evaluated["restored"]
            baseline = evaluated["baseline"]
            score = validation_score(metrics)

            record = {
                "epoch": epoch,
                "train_reward": avg_reward,
                **metrics,
                "baseline_psnr": baseline["psnr"],
                "baseline_ssim": baseline["ssim"],
                "validation_score": score,
            }

            history.append(record)

            print_metrics("Validation: original input", baseline)
            print_metrics("Validation: restored output", metrics)

            print(
                f"PSNR change vs input: "
                f"{metrics['psnr'] - baseline['psnr']:+.4f} dB"
            )

            if score > best_val_score:
                best_val_score = score
                agent.save(args.best_checkpoint)
                print(f"Best checkpoint saved: {args.best_checkpoint}")

            save_history(args.history_file, history)
            print(f"History saved: {args.history_file}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            agent.save(args.checkpoint)
            print(f"Latest checkpoint saved: {args.checkpoint}")

    # =====================================================
    # FINAL TEST
    # =====================================================

    print("\nTraining complete.")

    best_agent = DDPGAgent(
        device=args.device,
        gamma=args.gamma,
        tau=args.tau,
    )
    best_agent.load(args.best_checkpoint)

    test_results = evaluate_agent(
        best_agent,
        test_pairs,
        max_steps=args.max_steps,
    )

    print_metrics("Final test: original input", test_results["baseline"])
    print_metrics("Final test: restored output", test_results["restored"])

    improvement = (
        test_results["restored"]["psnr"]
        - test_results["baseline"]["psnr"]
    )

    print(f"\nTest PSNR improvement: {improvement:+.4f} dB")

    if test_results["restored"]["psnr"] >= 20.0:
        print("Target reached: test PSNR is at least 20 dB.")
    else:
        print(
            "Target not reached: test PSNR is below 20 dB. "
            "Inspect the environment reward and restoration operations."
        )

    print(f"Best checkpoint : {args.best_checkpoint}")
    print(f"Latest checkpoint: {args.checkpoint}")
    print(f"Metric history  : {args.history_file}")


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DDPG for UIEB underwater image restoration"
    )

    parser.add_argument("--data-dir", default="data/UIEB")
    parser.add_argument("--raw-folder", default="raw")
    parser.add_argument("--ref-folder", default="reference")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--max-dim", type=int, default=256)

    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--noise-start", type=float, default=0.5)
    parser.add_argument("--noise-end", type=float, default=0.1)
    parser.add_argument("--noise-decay", type=float, default=0.9998)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ddpg_actor_critic.pt",
    )
    parser.add_argument(
        "--best-checkpoint",
        default="checkpoints/best_ddpg_actor_critic.pt",
    )
    parser.add_argument("--resume", default=None)

    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--history-file", default="training_history.json")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
