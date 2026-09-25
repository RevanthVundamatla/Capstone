const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadPrompt = document.getElementById('uploadPrompt');
const compareView = document.getElementById('compareView');
const imgBefore = document.getElementById('imgBefore');
const canvasAfter = document.getElementById('imgAfter');
const resetBtn = document.getElementById('resetBtn');

browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
  if (!compareView.hidden) return;
  fileInput.click();
});

['dragenter', 'dragover'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  })
);
dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});
resetBtn.addEventListener('click', resetDemo);

function handleFile(file) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  imgBefore.src = url;
  imgBefore.onload = () => {
    uploadPrompt.hidden = true;
    compareView.hidden = false;
    resetBtn.hidden = false;
    restoreWithModel(imgBefore, canvasAfter);
    animateMetrics();
  };
}

function resetDemo() {
  compareView.hidden = true;
  uploadPrompt.hidden = false;
  resetBtn.hidden = true;
  fileInput.value = '';
  document.querySelectorAll('.metric-fill').forEach(el => el.style.width = '0%');
  document.querySelectorAll('.metric-val').forEach(el => el.textContent = '—');
}

// Placeholder for the trained DDPG policy's inference call.
// Swap this out for a fetch() to your real restoration endpoint —
// send the image, receive corrected pixels or a data URL back.
function restoreWithModel(sourceImg, targetCanvas) {
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight;
  targetCanvas.width = w;
  targetCanvas.height = h;
  const ctx = targetCanvas.getContext('2d');
  ctx.drawImage(sourceImg, 0, 0, w, h);

  const frame = ctx.getImageData(0, 0, w, h);
  const d = frame.data;
  // Simple stand-in "correction": counter underwater color cast by
  // lifting red/green relative to blue, and stretching contrast.
  for (let i = 0; i < d.length; i += 4) {
    d[i]     = clamp((d[i]     - 128) * 1.15 + 128 + 18); // R
    d[i + 1] = clamp((d[i + 1] - 128) * 1.12 + 128 + 8);  // G
    d[i + 2] = clamp((d[i + 2] - 128) * 1.05 + 128 - 10); // B
  }
  ctx.putImageData(frame, 0, 0);
}

function clamp(v) {
  return Math.max(0, Math.min(255, v));
}

// Illustrative score ranges per metric, animated in on reveal.
const METRIC_RANGES = {
  psnr:  { min: 21, max: 27, decimals: 1, suffix: ' dB' },
  ssim:  { min: 0.78, max: 0.91, decimals: 2, suffix: '' },
  uiqm:  { min: 2.6, max: 3.4, decimals: 2, suffix: '' },
  uciqe: { min: 0.52, max: 0.63, decimals: 2, suffix: '' },
};

function animateMetrics() {
  document.querySelectorAll('.metric-row').forEach(row => {
    const key = row.dataset.metric;
    const range = METRIC_RANGES[key];
    const value = range.min + Math.random() * (range.max - range.min);
    const pct = Math.round(((value - range.min) / (range.max - range.min)) * 55 + 35);
    const fill = row.querySelector('.metric-fill');
    const val = row.querySelector('.metric-val');
    requestAnimationFrame(() => {
      fill.style.width = pct + '%';
    });
    val.textContent = value.toFixed(range.decimals) + range.suffix;
  });
}
