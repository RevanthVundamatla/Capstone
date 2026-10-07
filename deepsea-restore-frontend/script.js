```javascript
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadPrompt = document.getElementById('uploadPrompt');
const compareView = document.getElementById('compareView');
const imgBefore = document.getElementById('imgBefore');
const canvasAfter = document.getElementById('imgAfter');
const resetBtn = document.getElementById('resetBtn');

// Your Render backend
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

['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
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

    // Clear old metrics
    document.querySelectorAll('.metric-fill').forEach(el => {
      el.style.width = '0%';
    });

    document.querySelectorAll('.metric-val').forEach(el => {
      el.textContent = '—';
    });

    // Send image to the real DDPG backend
    restoreWithModel(file, canvasAfter);
  };
}


function resetDemo() {
  compareView.hidden = true;
  uploadPrompt.hidden = false;
  resetBtn.hidden = true;

  fileInput.value = '';

  document.querySelectorAll('.metric-fill').forEach(el => {
    el.style.width = '0%';
  });

  document.querySelectorAll('.metric-val').forEach(el => {
    el.textContent = '—';
  });

  canvasAfter.width = 0;
  canvasAfter.height = 0;
}


async function restoreWithModel(file, targetCanvas) {
  try {
    // Show loading message
    document.querySelectorAll('.metric-val').forEach(el => {
      el.textContent = '...';
    });

    console.log('Sending image to backend...');

    // Create FormData
    const formData = new FormData();

    // Backend expects the uploaded image as "image"
    formData.append('image', file, file.name);

    // Call the Render backend
    const response = await fetch(`${API_URL}/api/restore`, {
      method: 'POST',
      body: formData
    });

    console.log('Backend status:', response.status);

    // Read response
    const result = await response.json();

    console.log('Backend response:', result);

    if (!response.ok) {
      throw new Error(
        result.error ||
        result.message ||
        `Backend returned HTTP ${response.status}`
      );
    }

    // Check restored image
    if (!result.image) {
      throw new Error('Backend did not return a restored image.');
    }

    // Display restored image
    displayRestoredImage(result.image, targetCanvas);

    // Display real metrics returned by backend
    if (result.metrics) {
      updateMetrics(result.metrics);
    }

    console.log('Restoration completed successfully.');

  } catch (error) {
    console.error('Restoration failed:', error);

    // Reset metric display
    document.querySelectorAll('.metric-val').forEach(el => {
      el.textContent = '—';
    });

    document.querySelectorAll('.metric-fill').forEach(el => {
      el.style.width = '0%';
    });

    alert(
      'Image restoration failed.\n\n' +
      error.message +
      '\n\nPlease check the backend and try again.'
    );
  }
}


function displayRestoredImage(imageData, targetCanvas) {
  const restoredImage = new Image();

  restoredImage.onload = () => {
    targetCanvas.width = restoredImage.naturalWidth;
    targetCanvas.height = restoredImage.naturalHeight;

    const ctx = targetCanvas.getContext('2d');

    ctx.clearRect(
      0,
      0,
      targetCanvas.width,
      targetCanvas.height
    );

    ctx.drawImage(
      restoredImage,
      0,
      0,
      targetCanvas.width,
      targetCanvas.height
    );
  };

  // Backend returns a data URL.
  // If it returns only Base64, add the PNG prefix.
  if (imageData.startsWith('data:image')) {
    restoredImage.src = imageData;
  } else {
    restoredImage.src = `data:image/png;base64,${imageData}`;
  }
}


function updateMetrics(metrics) {
  console.log('Metrics:', metrics);

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
    const row = document.querySelector(
      `.metric-row[data-metric="${key}"]`
    );

    if (!row) {
      return;
    }

    if (metrics[key] === undefined || metrics[key] === null) {
      return;
    }

    const valueElement = row.querySelector('.metric-val');
    const fillElement = row.querySelector('.metric-fill');

    const value = Number(metrics[key]);

    if (Number.isNaN(value)) {
      valueElement.textContent = '—';
      return;
    }

    // Display metric value
    valueElement.textContent =
      value.toFixed(metricConfig[key].decimals) +
      metricConfig[key].suffix;

    // Calculate progress-bar percentage
    let percentage = 0;

    if (key === 'psnr') {
      // Typical PSNR range for visualization
      percentage = (value / 40) * 100;

    } else if (key === 'ssim') {
      // SSIM normally ranges from 0 to 1
      percentage = value * 100;

    } else if (key === 'uiqm') {
      // Visualization scale
      percentage = (value / 5) * 100;

    } else if (key === 'uciqe') {
      // UCIQE normally around 0-1
      percentage = value * 100;
    }

    percentage = Math.max(
      0,
      Math.min(100, percentage)
    );

    requestAnimationFrame(() => {
      fillElement.style.width = `${percentage}%`;
    });
  });
}
```
