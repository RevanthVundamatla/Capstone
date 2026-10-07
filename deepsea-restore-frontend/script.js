const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadPrompt = document.getElementById('uploadPrompt');
const compareView = document.getElementById('compareView');
const imgBefore = document.getElementById('imgBefore');
const canvasAfter = document.getElementById('imgAfter');
const resetBtn = document.getElementById('resetBtn');

// Render backend URL
const API_URL = "https://capstone-deepsea-restore-backend.onrender.com";

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


function handleFile(file) {

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

  clearMetrics();

  canvasAfter.width = 0;
  canvasAfter.height = 0;
}


function clearMetrics() {

  document.querySelectorAll('.metric-fill').forEach((element) => {
    element.style.width = '0%';
  });

  document.querySelectorAll('.metric-val').forEach((element) => {
    element.textContent = '—';
  });
}


async function restoreWithModel(file, targetCanvas) {

  try {

    document.querySelectorAll('.metric-val').forEach((element) => {
      element.textContent = '...';
    });

    console.log('Sending image to DeepSea Restore backend...');

    const formData = new FormData();

    // Backend expects the uploaded image as "image"
    formData.append('image', file, file.name);

    // Call Render backend
    const response = await fetch(
      `${API_URL}/api/restore`,
      {
        method: 'POST',
        body: formData
      }
    );

    console.log('Backend HTTP status:', response.status);

    let result;

    try {
      result = await response.json();
    } catch (jsonError) {
      throw new Error(
        `Backend returned an invalid response. HTTP ${response.status}`
      );
    }

    console.log('Backend response:', result);

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
    displayRestoredImage(
      result.image,
      targetCanvas
    );

    // Display real metrics
    if (result.metrics) {

      updateMetrics(
        result.metrics
      );

    } else {

      console.warn(
        'No metrics were returned by the backend.'
      );
    }

    console.log(
      'Image restoration completed successfully.'
    );

  } catch (error) {

    console.error(
      'Image restoration failed:',
      error
    );

    clearMetrics();

    alert(
      'Image restoration failed.\n\n' +
      error.message +
      '\n\nPlease check the Render backend and try again.'
    );
  }
}


function displayRestoredImage(
  imageData,
  targetCanvas
) {

  const restoredImage = new Image();

  restoredImage.onload = () => {

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
  };

  restoredImage.onerror = () => {

    console.error(
      'Could not load restored image.'
    );

    alert(
      'The backend returned an invalid restored image.'
    );
  };

  // Backend returns a complete data URL
  if (
    typeof imageData === 'string' &&
    imageData.startsWith('data:image')
  ) {

    restoredImage.src = imageData;

  } else {

    // Backend returns raw Base64
    restoredImage.src =
      `data:image/png;base64,${imageData}`;
  }
}


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

    if (
      metrics[key] === undefined ||
      metrics[key] === null
    ) {
      return;
    }

    const valueElement =
      row.querySelector('.metric-val');

    const fillElement =
      row.querySelector('.metric-fill');

    const value =
      Number(metrics[key]);

    if (Number.isNaN(value)) {

      valueElement.textContent = '—';

      return;
    }

    // Display metric value
    valueElement.textContent =
      value.toFixed(
        metricConfig[key].decimals
      ) +
      metricConfig[key].suffix;

    // Progress bar percentage
    let percentage = 0;

    if (key === 'psnr') {

      percentage =
        (value / 40) * 100;

    } else if (key === 'ssim') {

      percentage =
        value * 100;

    } else if (key === 'uiqm') {

      percentage =
        (value / 5) * 100;

    } else if (key === 'uciqe') {

      percentage =
        value * 100;
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
