# DeepSea Restore — DDPG Backend

Backend for **Reinforcement Learning-Based Underwater Degradation Image
Restoration** (Capstone Review-0, DDPG). Implements the architecture from
the review deck: a continuous-action Actor–Critic agent that applies
white-balance, gamma, contrast, saturation and dehaze corrections to
underwater images, trained on UIEB with a hybrid PSNR/SSIM/UIQM/UCIQE
reward.

## Project layout

```
app.py                     Flask API the frontend calls
inference.py                Runs the agent's multi-pass restoration on one image
train.py                    DDPG training loop over UIEB
ddpg/
  networks.py                Actor mu(s|theta), Critic Q(s,a|phi), shared state encoder
  agent.py                    Target networks, soft updates, train_step, checkpointing
  replay_buffer.py            (s, a, r, s', done) replay buffer
enhancement/
  ops.py                       Applies the action vector: WB, gamma, contrast, saturation, dehaze
  state.py                     Image -> compact state tensor for the encoder
  environment.py               The MDP: reset/step, hybrid reward function
metrics/
  metrics.py                   PSNR, SSIM (skimage) + UIQM, UCIQE (implemented from their papers)
frontend_integration/
  script.js                    Drop-in replacement for the frontend's script.js
checkpoints/                  Trained weights land here (ddpg_actor_critic.pt)
data/                          Put the UIEB dataset here
```

## 1. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download the raw and reference splits, published separately on Kaggle:

- Raw (degraded):  https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw
- Reference (ground truth): https://www.kaggle.com/datasets/larjeck/uieb-dataset-reference

```bash
kaggle datasets download -d larjeck/uieb-dataset-raw -p data/UIEB/raw --unzip
kaggle datasets download -d larjeck/uieb-dataset-reference -p data/UIEB/reference --unzip
```

(Or download the zips from the Kaggle UI and unzip them into
`data/UIEB/raw` and `data/UIEB/reference` yourself.)

```
data/UIEB/raw/...
data/UIEB/reference/...
```

`train.py` auto-descends into a single nested folder if Kaggle's zip added
one extra directory level, and pairs images by matching filename first —
since these are two independently-published datasets rather than one
paired archive, if filenames don't line up it falls back to pairing by
sorted order as long as both folders have the same image count. Check the
printed pairing count at the start of training; if it looks wrong,
rename files so raw/reference pairs share a filename.

## 3. Train

```bash
python train.py --data-dir data/UIEB --episodes 20000
```

- Each episode picks one raw/reference pair and lets the agent apply up to
  `--max-steps` (default 3) correction passes, matching the deck's "either
  multiple passes on one image, or across the dataset" workflow step.
- Reward blends PSNR/SSIM against the reference with UIQM/UCIQE on the
  output, weighted 0.35/0.35/0.15/0.15 (see `enhancement/environment.py`
  if you want to change the balance).
- Checkpoints save to `checkpoints/ddpg_actor_critic.pt` every
  `--save-every` episodes. Resume with `--resume checkpoints/ddpg_actor_critic.pt`.
- Expect this to take real wall-clock time on CPU — pass `--device cuda`
  if you have a GPU.

## 4. Run the API

```bash
python app.py
# or: CHECKPOINT=checkpoints/ddpg_actor_critic.pt PORT=5000 python app.py
```

Without a checkpoint present, it still starts — the Actor is just
randomly initialized, so restorations won't be meaningful until you train.

`GET /api/health` — reports whether a trained checkpoint is loaded.

`POST /api/restore` — multipart form field `image`, or JSON
`{"image": "data:image/png;base64,..."}`. Returns:

```json
{
  "image": "data:image/png;base64,...",
  "metrics": { "psnr": 23.1, "ssim": 0.84, "uiqm": 3.05, "uciqe": 0.58 },
  "steps": [[...], [...], [...]],
  "checkpoint_loaded": true
}
```

## 5. Wire it into the frontend

Your frontend repo's `script.js` currently fakes restoration with a canvas
filter. Replace it with `frontend_integration/script.js` from this backend
(same DOM element IDs, so no HTML changes needed), then either:

- serve the frontend from a different origin and set
  `window.DEEPSEA_API_BASE = "http://your-api-host:5000"` before
  `script.js` loads, or
- leave it as `http://localhost:5000` for local development.

Also enable CORS for your real frontend origin before deploying — `app.py`
defaults `ALLOWED_ORIGINS` to `*` for convenience during development:

```bash
ALLOWED_ORIGINS=https://your-frontend-domain python app.py
```

## Notes on the implementation

- **Why continuous actions:** matches the deck's core argument — gain,
  gamma and dehaze strength are real-valued, so a discrete filter bank
  forces banding. The Actor outputs all 7 parameters directly via `tanh`.
- **Dehaze:** `enhancement/ops.py` uses a dark-channel-prior estimate
  (He et al.-style), blended by the agent's `dehaze_strength` action —
  a reasonable stand-in for a differentiable dehaze block, not a learned one.
- **UIQM / UCIQE:** implemented directly from Panetta et al. (2016) and
  Yang & Sowmya (2015) respectively, since neither ships in skimage.
- **Reward-metric misalignment:** deliberately weighted so no-reference
  scores (UIQM/UCIQE) can't dominate the reward and produce the
  over-saturated, ground-truth-unfaithful outputs the deck's limitations
  slide warns about.
