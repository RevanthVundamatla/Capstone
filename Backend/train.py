"""
DDPG training for underwater image restoration on the UIEB dataset.

Dataset:
    data/UIEB/raw
    data/UIEB/reference

The dataset is split into:
    80% training
    10% validation
    10% test

Training uses only the training split.

The best checkpoint is selected using validation PSNR + SSIM.

Final evaluation reports:
    PSNR
    SSIM
    UIQM
    UCIQE
"""

from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np

from ddpg.agent import DDPGAgent
from ddpg.replay_buffer import ReplayBuffer
from enhancement.environment import UnderwaterEnhanceEnv
from metrics.metrics import psnr, ssim, uiqm, uciqe


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32) / 255.0


# =========================================================
# DATASET DIRECTORY RESOLUTION
# =========================================================

def _resolve_image_dir(path: str) -> str:

    entries = os.listdir(path)

    if any(f.lower().endswith(IMG_EXTS) for f in entries):
        return path

    subdirs = [
        d for d in entries
        if os.path.isdir(os.path.join(path, d))
    ]

    if len(subdirs) == 1:
        return _resolve_image_dir(
            os.path.join(path, subdirs[0])
        )

    return path


# =========================================================
# DATASET PAIRING
# =========================================================

def list_pairs(
    data_dir: str,
    raw_folder="raw",
    ref_folder="reference"
):

    raw_dir = _resolve_image_dir(
        os.path.join(data_dir, raw_folder)
    )

    ref_dir = _resolve_image_dir(
        os.path.join(data_dir, ref_folder)
    )

    raw_names = sorted(
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(IMG_EXTS)
    )

    ref_names = sorted(
        f for f in os.listdir(ref_dir)
        if f.lower().endswith(IMG_EXTS)
    )

    if not raw_names or not ref_names:
        raise FileNotFoundError(
            f"No images found under:\n"
            f"Raw: {raw_dir}\n"
            f"Reference: {ref_dir}"
        )

    ref_by_name = {
        name: name
        for name in ref_names
    }

    pairs = []

    for name in raw_names:

        if name in ref_by_name:

            pairs.append(
                (
                    os.path.join(raw_dir, name),
                    os.path.join(ref_dir, name)
                )
            )

    if not pairs:

        if len(raw_names) == len(ref_names):

            print(
                f"No filename matches found.\n"
                f"Using sorted-order pairing for "
                f"{len(raw_names)} images."
            )

            pairs = [
                (
                    os.path.join(raw_dir, raw_name),
                    os.path.join(ref_dir, ref_name)
                )
                for raw_name, ref_name
                in zip(raw_names, ref_names)
            ]

        else:

            raise FileNotFoundError(
                f"Could not pair images.\n"
                f"Raw images: {len(raw_names)}\n"
                f"Reference images: {len(ref_names)}"
            )

    else:

        print(
            f"Paired {len(pairs)} images "
            f"by matching filename."
        )

    return pairs


# =========================================================
# DATASET SPLIT
# =========================================================

def split_dataset(
    pairs,
    train_ratio=0.80,
    val_ratio=0.10,
    seed=42
):

    pairs = list(pairs)

    rng = random.Random(seed)

    rng.shuffle(pairs)

    total = len(pairs)

    train_end = int(total * train_ratio)

    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]

    val_pairs = pairs[train_end:val_end]

    test_pairs = pairs[val_end:]

    return train_pairs, val_pairs, test_pairs


# =========================================================
# METRIC EVALUATION
# =========================================================

def evaluate_agent(
    agent,
    pairs,
    max_steps=3
):

    env = UnderwaterEnhanceEnv(
        max_steps=max_steps
    )

    psnr_values = []
    ssim_values = []
    uiqm_values = []
    uciqe_values = []

    for raw_path, ref_path in pairs:

        raw_img = load_image(raw_path)
        ref_img = load_image(ref_path)

        state = env.reset(
            raw_img,
            ref_img
        )

        done = False

        while not done:

            action = agent.select_action(
                state,
                noise_std=0.0
            )

            state, reward, done, info = env.step(
                action
            )

        restored = env.current

        psnr_values.append(
            psnr(restored, ref_img)
        )

        ssim_values.append(
            ssim(restored, ref_img)
        )

        uiqm_values.append(
            uiqm(restored)
        )

        uciqe_values.append(
            uciqe(restored)
        )

    results = {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "uiqm": float(np.mean(uiqm_values)),
        "uciqe": float(np.mean(uciqe_values)),
    }

    return results


# =========================================================
# TRAINING
# =========================================================

def train(args):

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    pairs = list_pairs(
        args.data_dir,
        args.raw_folder,
        args.ref_folder
    )

    print(
        f"Loaded {len(pairs)} raw/reference pairs "
        f"from {args.data_dir}"
    )

    # -----------------------------------------------------
    # Split dataset
    # -----------------------------------------------------

    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs,
        train_ratio=0.80,
        val_ratio=0.10,
        seed=args.seed
    )

    print()
    print("Dataset split")
    print("-----------------------------")
    print(f"Training   : {len(train_pairs)}")
    print(f"Validation : {len(val_pairs)}")
    print(f"Test       : {len(test_pairs)}")
    print("-----------------------------")
    print()

    # -----------------------------------------------------
    # Create agent
    # -----------------------------------------------------

    agent = DDPGAgent(
        device=args.device,
        gamma=args.gamma,
        tau=args.tau
    )

    if args.resume and os.path.exists(args.resume):

        agent.load(args.resume)

        print(
            f"Resumed model from: {args.resume}"
        )

    # -----------------------------------------------------
    # Replay buffer
    # -----------------------------------------------------

    buffer = ReplayBuffer(
        capacity=args.buffer_size
    )

    # -----------------------------------------------------
    # Environment
    # -----------------------------------------------------

    env = UnderwaterEnhanceEnv(
        max_steps=args.max_steps
    )

    # -----------------------------------------------------
    # Training variables
    # -----------------------------------------------------

    noise_std = args.noise_start

    reward_history = []

    best_val_psnr = -float("inf")

    best_val_ssim = -float("inf")

    os.makedirs(
        os.path.dirname(args.checkpoint),
        exist_ok=True
    )

    best_checkpoint = args.best_checkpoint

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for episode in range(
        1,
        args.episodes + 1
    ):

        raw_path, ref_path = random.choice(
            train_pairs
        )

        raw_img = load_image(
            raw_path
        )

        ref_img = load_image(
            ref_path
        )

        state = env.reset(
            raw_img,
            ref_img
        )

        episode_reward = 0.0

        info = {}

        # -------------------------------------------------
        # Episode
        # -------------------------------------------------

        for _ in range(args.max_steps):

            action = agent.select_action(
                state,
                noise_std=noise_std
            )

            (
                next_state,
                reward,
                done,
                info
            ) = env.step(action)

            buffer.push(
                state,
                action,
                reward,
                next_state,
                float(done)
            )

            state = next_state

            episode_reward += reward

            # ---------------------------------------------
            # DDPG update
            # ---------------------------------------------

            agent.train_step(
                buffer,
                batch_size=args.batch_size
            )

            if done:
                break

        # -------------------------------------------------
        # Noise decay
        # -------------------------------------------------

        noise_std = max(
            args.noise_end,
            noise_std * args.noise_decay
        )

        reward_history.append(
            episode_reward
        )

        # =================================================
        # LOGGING
        # =================================================

        if episode % args.log_every == 0:

            avg_reward = np.mean(
                reward_history[
                    -args.log_every:
                ]
            )

            print(
                f"episode {episode:6d}  "
                f"avg_reward={avg_reward:.4f}  "
                f"noise_std={noise_std:.3f}  "
                f"buffer={len(buffer)}  "
                f"psnr={info.get('psnr', float('nan')):.2f}  "
                f"ssim={info.get('ssim', float('nan')):.3f}  "
                f"uiqm={info.get('uiqm', float('nan')):.3f}  "
                f"uciqe={info.get('uciqe', float('nan')):.3f}"
            )

        # =================================================
        # VALIDATION
        # =================================================

        if episode % args.eval_every == 0:

            print()
            print(
                f"Evaluating validation set "
                f"at episode {episode}..."
            )

            val_results = evaluate_agent(
                agent,
                val_pairs,
                args.max_steps
            )

            print(
                f"Validation -> "
                f"PSNR: {val_results['psnr']:.3f} | "
                f"SSIM: {val_results['ssim']:.4f} | "
                f"UIQM: {val_results['uiqm']:.4f} | "
                f"UCIQE: {val_results['uciqe']:.4f}"
            )

            # -------------------------------------------------
            # Best model selection
            #
            # PSNR is primary.
            # SSIM is used as a tie-breaker.
            # -------------------------------------------------

            is_better = False

            if val_results["psnr"] > best_val_psnr:

                is_better = True

            elif (
                abs(
                    val_results["psnr"]
                    - best_val_psnr
                ) < 1e-6
                and val_results["ssim"]
                > best_val_ssim
            ):

                is_better = True

            if is_better:

                best_val_psnr = val_results["psnr"]

                best_val_ssim = val_results["ssim"]

                agent.save(
                    best_checkpoint
                )

                print(
                    f"  BEST MODEL SAVED -> "
                    f"{best_checkpoint}"
                )

            print()

        # =================================================
        # PERIODIC CHECKPOINT
        # =================================================

        if episode % args.save_every == 0:

            agent.save(
                args.checkpoint
            )

            print(
                f"  checkpoint saved -> "
                f"{args.checkpoint}"
            )


    # =====================================================
    # FINAL MODEL
    # =====================================================

    agent.save(
        args.checkpoint
    )

    print()
    print(
        f"Training complete."
    )

    print(
        f"Latest checkpoint -> "
        f"{args.checkpoint}"
    )

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    if os.path.exists(best_checkpoint):

        print()
        print(
            f"Loading best validation model:"
        )

        print(
            best_checkpoint
        )

        best_agent = DDPGAgent(
            device=args.device,
            gamma=args.gamma,
            tau=args.tau
        )

        best_agent.load(
            best_checkpoint
        )

        # =================================================
        # FINAL TEST
        # =================================================

        print()
        print(
            "Final test evaluation"
        )

        print(
            "================================"
        )

        test_results = evaluate_agent(
            best_agent,
            test_pairs,
            args.max_steps
        )

        print(
            f"PSNR  : {test_results['psnr']:.4f} dB"
        )

        print(
            f"SSIM  : {test_results['ssim']:.4f}"
        )

        print(
            f"UIQM  : {test_results['uiqm']:.4f}"
        )

        print(
            f"UCIQE : {test_results['uciqe']:.4f}"
        )

        print(
            "================================"
        )

        print()
        print(
            "Best model is ready for inference."
        )


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-dir",
        default="data/UIEB"
    )

    p.add_argument(
        "--raw-folder",
        default="raw"
    )

    p.add_argument(
        "--ref-folder",
        default="reference"
    )

    p.add_argument(
        "--episodes",
        type=int,
        default=20000
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=3
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    p.add_argument(
        "--buffer-size",
        type=int,
        default=100_000
    )

    p.add_argument(
        "--gamma",
        type=float,
        default=0.9
    )

    p.add_argument(
        "--tau",
        type=float,
        default=0.005
    )

    p.add_argument(
        "--noise-start",
        type=float,
        default=0.3
    )

    p.add_argument(
        "--noise-end",
        type=float,
        default=0.02
    )

    p.add_argument(
        "--noise-decay",
        type=float,
        default=0.9995
    )

    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"]
    )

    p.add_argument(
        "--checkpoint",
        default="checkpoints/ddpg_actor_critic.pt"
    )

    p.add_argument(
        "--best-checkpoint",
        default="checkpoints/best_ddpg_actor_critic.pt"
    )

    p.add_argument(
        "--resume",
        default=None
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=50
    )

    p.add_argument(
        "--eval-every",
        type=int,
        default=500
    )

    p.add_argument(
        "--save-every",
        type=int,
        default=500
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return p.parse_args()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args)