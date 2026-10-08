const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadPrompt = document.getElementById('uploadPrompt');
const compareView = document.getElementById('compareView');
const imgBefore = document.getElementById('imgBefore');
const canvasAfter = document.getElementById('imgAfter');
const resetBtn = document.getElementById('resetBtn');

// Optional reference image (enables PSNR and SSIM on the backend)
const refInput = document.getElementById('refInput');

// Render backend URL
const API_URL = "https://capstone-deepsea-restore-backend.onrender.com";

// Max time to wait for the backend (ms). Render free tier can be slow.
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

// Bar scales for the no-reference metrics.
// Keep these in sync with UIQM_NORM and UCIQE_NORM in environment.py.
const UIQM_BAR_MAX = 6;
const UCIQE_BAR_MAX = 12;

// Download button + status message
let downloadBtn = null;
let statusEl = null;
let isProcessing = false;


/* ------------------------------------------------------------------ */
/* Wake the backend as soon as the page loads (Render free tier sleeps) */
/* ------------------------------------------------------------------ */

(function wakeBackend() {
  fetch(`${API_URL}/api/health`)
    .then((r) => r.json())
    .then((data) => console.log('Backend health:', data))
    .catch((err) => console.warn('Backend wake-up ping failed:', err));
})();


/* ------------------------------------------------------------------ */
/* Status message helpers                                              */
/* ------------------------------------------------------------------ */

function showStatus(message) {
  if (!statusEl) {
    statusEl = document.createElement('div');
    statusEl.style.margin = '12px 0';
    statusEl.style.fontSize = '14px';
    statusEl.style.opacity = '0.85';
    compareView.parentNode.insertBefore(statusEl, compareView.nextSibling);
  }
  statusEl.textContent = message;
  statusEl.hidden = false;
}

function hideStatus() {
  if (statusEl) {
    statusEl.hidden = true;
    statusEl.textContent = '';
  }
}


/* ------------------------------------------------------------------ */
/* UI events                                                           */
/* ------------------------------------------------------------------ */

browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener('click', (e) => {
  if (!compareView.hidden) return;

  if (e.target === browseBtn) return;

  fileInput.click();
});

['dragenter', 'dragover'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();

    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();

    dropZone.classList.remove('dragover');
  });
});

dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];

  if (file) {
    handleFile(file);
  }
});

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];

  if (file) {
    handleFile(file);
  }
});

resetBtn.addEventListener('click', resetDemo);


/* ------------------------------------------------------------------ */
/* File handling                                                       */
/* ------------------------------------------------------------------ */

function handleFile(file) {

  if (isProcessing) {
    alert('Please wait until the current image finishes processing.');
    return;
  }

  if (!file.type.startsWith('image/')) {
    alert('Please select a JPG or PNG image.');
    return;
  }

  const url = URL.createObjectURL(file);

  imgBefore.src = url;

  imgBefore.onload = () => {

    uploadPrompt.hidden = true;
    compareView.hidden = false;
    resetBtn.hidden = false;

    clearMetrics();

    restoreWithModel(file, canvasAfter);
  };
}


function resetDemo() {

  compareView.hidden = true;
  uploadPrompt.hidden = false;
  resetBtn.hidden = true;

  fileInput.value = '';

  // Clear the optional reference so it is not reused by accident
  if (refInput) {
    refInput.value = '';
  }

  clearMetrics();
  hideStatus();

  canvasAfter.width = 0;
  canvasAfter.height = 0;

  if (downloadBtn) {
    downloadBtn.remove();
    downloadBtn = null;
  }
}


function clearMetrics() {

  document.querySelectorAll('.metric-fill').forEach((element) => {
    element.style.width = '0%';
  });

  document.querySelectorAll('.metric-val').forEach((element) => {
    element.textContent = '—';
  });
}


/* ------------------------------------------------------------------ */
/* Backend request                                                     */
/* ------------------------------------------------------------------ */

async function postImage(file) {

  const formData = new FormData();

  // Backend expects the uploaded image as "image"
  formData.append('image', file, file.name);

  // Optional reference image: lets the backend compute PSNR and SSIM
  if (refInput && refInput.files[0]) {
    formData.append(
      'reference',
      refInput.files[0],
      refInput.files[0].name
    );
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    return await fetch(
      `${API_URL}/api/restore`,
      {
        method: 'POST',
        body: formData,
        signal: controller.signal
      }
    );
  } finally {
    clearTimeout(timeoutId);
  }
}


async function restoreWithModel(file, targetCanvas) {

  isProcessing = true;

  try {

    document.querySelectorAll('.metric-val').forEach((element) => {
      element.textContent = '...';
    });

    showStatus(
      'Restoring image... the server may need up to a minute to wake up and process.'
    );

    console.log('Sending image to DeepSea Restore backend...');
    console.log('API URL:', API_URL);
    console.log('Sending file:', file.name);
    console.log('File type:', file.type);
    console.log('File size:', file.size);
    console.log(
      'Reference attached:',
      Boolean(refInput && refInput.files[0])
    );

    let response;

    try {
      response = await postImage(file);
    } catch (firstError) {

      // Cold start / dropped connection: retry once
      if (firstError.name === 'AbortError') {
        throw firstError;
      }

      console.warn('First request failed, retrying once...', firstError);

      showStatus('Server is waking up, retrying...');

      await new Promise((resolve) => setTimeout(resolve, 5000));

      response = await postImage(file);
    }

    console.log('Backend HTTP status:', response.status);

    const responseText = await response.text();

    let result;

    try {
      result = JSON.parse(responseText);
    } catch (jsonError) {
      console.error('Raw backend response:', responseText);

      throw new Error(
        `Backend returned invalid JSON. HTTP ${response.status}`
      );
    }

    console.log('Backend response received.');

    if (!response.ok) {

      throw new Error(
        result.error ||
        result.message ||
        `Backend returned HTTP ${response.status}`
      );
    }

    if (!result.image) {

      throw new Error(
        'Backend response does not contain a restored image.'
      );
    }

    // Display restored image
    await displayRestoredImage(
      result.image,
      targetCanvas
    );

    // Display metrics
    if (result.metrics) {

      updateMetrics(
        result.metrics
      );

    } else {

      console.warn(
        'No metrics were returned by the backend.'
      );
    }

    // Create download button
    createDownloadButton(
      result.image
    );

    hideStatus();

    console.log(
      'Image restoration completed successfully.'
    );

  } catch (error) {

    console.error(
      'Image restoration failed:',
      error
    );

    clearMetrics();
    hideStatus();

    let message = error.message;

    if (error.name === 'AbortError') {
      message =
        'The request took too long and was cancelled. ' +
        'Try a smaller image or try again in a minute.';
    } else if (error instanceof TypeError) {
      message =
        'Could not reach the backend. It may be starting up or ' +
        'overloaded. Please wait a minute and try again.';
    }

    alert(
      'Image restoration failed.\n\n' +
      message +
      '\n\nOpen the browser Console (F12) for more details.'
    );

  } finally {

    isProcessing = false;
  }
}


/* ------------------------------------------------------------------ */
/* Display restored image                                              */
/* ------------------------------------------------------------------ */

function displayRestoredImage(
  imageData,
  targetCanvas
) {

  return new Promise((resolve, reject) => {

    const restoredImage = new Image();

    restoredImage.onload = () => {

      console.log(
        'Restored image loaded:',
        restoredImage.naturalWidth,
        'x',
        restoredImage.naturalHeight
      );

      targetCanvas.width =
        restoredImage.naturalWidth;

      targetCanvas.height =
        restoredImage.naturalHeight;

      const context =
        targetCanvas.getContext('2d');

      context.clearRect(
        0,
        0,
        targetCanvas.width,
        targetCanvas.height
      );

      context.drawImage(
        restoredImage,
        0,
        0,
        targetCanvas.width,
        targetCanvas.height
      );

      resolve();
    };

    restoredImage.onerror = (error) => {

      console.error(
        'Could not load restored image:',
        error
      );

      reject(
        new Error(
          'The backend returned an invalid restored image.'
        )
      );
    };

    // Backend returns a complete PNG data URL
    if (
      typeof imageData === 'string' &&
      imageData.startsWith('data:image')
    ) {

      restoredImage.src = imageData;

    } else {

      // Fallback for raw Base64
      restoredImage.src =
        `data:image/png;base64,${imageData}`;
    }
  });
}


/* ------------------------------------------------------------------ */
/* Download button                                                     */
/* ------------------------------------------------------------------ */

function createDownloadButton(imageData) {

  if (downloadBtn) {
    downloadBtn.remove();
  }

  downloadBtn = document.createElement('a');

  downloadBtn.textContent =
    'Download restored image';

  downloadBtn.className =
    'reset-btn';

  downloadBtn.style.display =
    'inline-block';

  downloadBtn.style.textDecoration =
    'none';

  downloadBtn.style.marginLeft =
    '10px';

  downloadBtn.href =
    imageData;

  downloadBtn.download =
    'deepsea-restored.png';

  resetBtn.parentNode.appendChild(
    downloadBtn
  );
}


/* ------------------------------------------------------------------ */
/* Metrics display                                                     */
/* ------------------------------------------------------------------ */

function updateMetrics(metrics) {

  console.log(
    'Restoration metrics:',
    metrics
  );

  const metricConfig = {

    psnr: {
      decimals: 2,
      suffix: ' dB'
    },

    ssim: {
      decimals: 4,
      suffix: ''
    },

    uiqm: {
      decimals: 4,
      suffix: ''
    },

    uciqe: {
      decimals: 4,
      suffix: ''
    }

  };

  Object.keys(metricConfig).forEach((key) => {

    const row =
      document.querySelector(
        `.metric-row[data-metric="${key}"]`
      );

    if (!row) {
      return;
    }

    const valueElement =
      row.querySelector('.metric-val');

    const fillElement =
      row.querySelector('.metric-fill');

    // PSNR and SSIM need a reference image; say so instead of "N/A"
    const missingText =
      (key === 'psnr' || key === 'ssim')
        ? 'needs reference'
        : 'N/A';

    if (
      metrics[key] === undefined ||
      metrics[key] === null
    ) {

      valueElement.textContent =
        missingText;

      fillElement.style.width =
        '0%';

      return;
    }

    const value =
      Number(metrics[key]);

    if (Number.isNaN(value)) {

      valueElement.textContent =
        missingText;

      fillElement.style.width =
        '0%';

      return;
    }

    valueElement.textContent =
      value.toFixed(
        metricConfig[key].decimals
      ) +
      metricConfig[key].suffix;

    let percentage = 0;

    if (key === 'psnr') {

      percentage =
        (value / 40) * 100;

    } else if (key === 'ssim') {

      percentage =
        value * 100;

    } else if (key === 'uiqm') {

      percentage =
        (value / UIQM_BAR_MAX) * 100;

    } else if (key === 'uciqe') {

      percentage =
        (value / UCIQE_BAR_MAX) * 100;
    }

    percentage =
      Math.max(
        0,
        Math.min(
          100,
          percentage
        )
      );

    requestAnimationFrame(() => {

      fillElement.style.width =
        `${percentage}%`;

    });

  });
}
