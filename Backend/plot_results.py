"""
Make training graphs from the saved training log.

Usage (from the Backend folder):
    python plot_results.py train_log.txt

Creates in ./figures/ :
    psnr_vs_episodes.png   validation PSNR vs training episode
    metric_curve.png       PSNR, SSIM, UIQM, UCIQE vs episode (validation)
    reward_curve.png       average training reward vs episode
"""

import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_path = sys.argv[1] if len(sys.argv) > 1 else "train_log.txt"

# PowerShell's Tee-Object may write UTF-16; try both encodings.
text = None
for enc in ("utf-8", "utf-16"):
    try:
        with open(log_path, encoding=enc) as f:
            text = f.read()
        if "episode" in text:
            break
    except (UnicodeError, FileNotFoundError):
        continue
if text is None or "episode" not in text:
    sys.exit(f"Could not read a training log from: {log_path}")

reward_ep, reward_val = [], []
val = {"ep": [], "psnr": [], "ssim": [], "uiqm": [], "uciqe": []}
current_ep = None

re_reward = re.compile(r"^episode\s+(\d+)\s+avg_reward=(-?[\d.]+)")
re_eval = re.compile(r"Evaluating validation set at episode (\d+)")
re_val = re.compile(
    r"Validation -> PSNR: ([\d.]+) \| SSIM: ([\d.]+) \| "
    r"UIQM: ([\d.]+) \| UCIQE: ([\d.]+)"
)

for line in text.splitlines():
    line = line.strip()
    m = re_reward.match(line)
    if m:
        reward_ep.append(int(m.group(1)))
        reward_val.append(float(m.group(2)))
        continue
    m = re_eval.search(line)
    if m:
        current_ep = int(m.group(1))
        continue
    m = re_val.search(line)
    if m and current_ep is not None:
        val["ep"].append(current_ep)
        for key, grp in zip(("psnr", "ssim", "uiqm", "uciqe"), m.groups()):
            val[key].append(float(grp))

if not val["ep"]:
    sys.exit("No validation lines found. Did training reach an --eval-every step?")

os.makedirs("figures", exist_ok=True)

# 1. PSNR vs episodes
plt.figure(figsize=(6, 4))
plt.plot(val["ep"], val["psnr"], marker="o")
plt.xlabel("Training episode")
plt.ylabel("Validation PSNR (dB)")
plt.title("Validation PSNR vs training episode")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/psnr_vs_episodes.png", dpi=200)
plt.close()

# 2. All four metrics
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (key, label) in zip(
    axes.ravel(),
    [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("uiqm", "UIQM"), ("uciqe", "UCIQE")],
):
    ax.plot(val["ep"], val[key], marker="o")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(label)
    ax.set_title(f"Validation {label}")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/metric_curve.png", dpi=200)
plt.close(fig)

# 3. Training reward
if reward_ep:
    plt.figure(figsize=(6, 4))
    plt.plot(reward_ep, reward_val)
    plt.xlabel("Training episode")
    plt.ylabel("Average episode reward")
    plt.title("Training reward vs episode")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/reward_curve.png", dpi=200)
    plt.close()

print("Saved figures to ./figures/ :", sorted(os.listdir("figures")))
print("Validation points:", len(val["ep"]), "| reward points:", len(reward_ep))
