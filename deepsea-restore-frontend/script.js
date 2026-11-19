
"use strict";

/* =========================================================
   CONFIGURATION
   ========================================================= */

// Local backend for development.
// For the deployed website, change this to your deployed
// Render backend URL.
const API_URL = "http://127.0.0.1:5000";

const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;
const UIQM_BAR_MAX = 6;
const UCIQE_BAR_MAX = 12;

/* =========================================================
   ELEMENTS
   ========================================================= */

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadPrompt = document.getElementById("uploadPrompt");
const compareView = document.getElementById("compareView");
const imgBefore = document.getElementById("imgBefore");
const canvasAfter = document.getElementById("imgAfter");
const resetBtn = document.getElementById("resetBtn");
const refInput = document.getElementById("refInput");
const downloadBtn = document.getElementById("downloadBtn");
const statusMessage = document.getElementById("statusMessage");

const historyStatus = document.getElementById("historyStatus");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");

/* =========================================================
   STATE
   ========================================================= */

let isProcessing = false;
let currentObjectUrl = null;
const trainingCharts = {};

/* =========================================================
   BACKEND HEALTH CHECK
   ========================================================= */

async function checkBackend() {
  try {
    const response = await fetch(`${API_URL}/api/health`);
    const data = await response.json();

    console.log("Backend health:", data);
  } catch (error) {
    console.warn("Backend health check failed:", error);
  }
}

checkBackend();

/* =========================================================
   STATUS
   ========================================================= */

function showStatus(message) {
  statusMessage.textContent = message;
}

function hideStatus() {
  statusMessage.textContent = "";
}

/* =========================================================
   UPLOAD EVENTS
   ========================================================= */

browseBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("click", (event) => {
  if (!compareView.hidden) return;
  if (event.target === browseBtn) return;

  fileInput.click();
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();

    if (compareView.hidden) {
      fileInput.click();
    }
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];

  if (file) handleFile(file);
});

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];

  if (file) handleFile(file);
});

resetBtn.addEventListener("click", resetDemo);

/* =========================================================
   FILE HANDLING
   ========================================================= */

function handleFile(file) {
  if (isProcessing) {
    alert("Please wait for the current image to finish processing.");
    return;
  }

  if (!file.type.startsWith("image/")) {
    alert("Please select a JPG, PNG, or other supported image.");
    return;
  }

  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
  }

  currentObjectUrl = URL.createObjectURL(file);
  imgBefore.src = currentObjectUrl;

  imgBefore.onload = () => {
    uploadPrompt.hidden = true;
    compareView.hidden = false;
    resetBtn.hidden = false;

    clearMetrics();
    canvasAfter.width = 0;
    canvasAfter.height = 0;

    restoreWithModel(file);
  };

  imgBefore.onerror = () => {
    showStatus("Could not read the selected image.");
  };
}

function resetDemo() {
  compareView.hidden = true;
  uploadPrompt.hidden = false;
  resetBtn.hidden = true;

  fileInput.value = "";
  refInput.value = "";

  imgBefore.removeAttribute("src");
  canvasAfter.width = 0;
  canvasAfter.height = 0;

  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }

  downloadBtn.hidden = true;
  downloadBtn.removeAttribute("href");

  clearMetrics();
  hideStatus();
}

function clearMetrics() {
  document.querySelectorAll(".metric-fill").forEach((element) => {
    element.style.width = "0%";
  });

  document.querySelectorAll(".metric-val").forEach((element) => {
    element.textContent = "—";
  });
}

/* =========================================================
   RESTORATION API REQUEST
   ========================================================= */

async function postImage(file, signal) {
  const formData = new FormData();

  formData.append("image", file, file.name);

  if (refInput.files && refInput.files[0]) {
    formData.append(
      "reference",
      refInput.files[0],
      refInput.files[0].name
    );
  }

  return fetch(`${API_URL}/api/restore`, {
    method: "POST",
    body: formData,
    signal
  });
}

async function restoreWithModel(file) {
  isProcessing = true;

  const controller = new AbortController();

  const timeoutId = setTimeout(
    () => controller.abort(),
    REQUEST_TIMEOUT_MS
  );

  try {
    document.querySelectorAll(".metric-val").forEach((element) => {
      element.textContent = "...";
    });

    showStatus("Sending image to the DDPG restoration backend...");

    console.log("API URL:", API_URL);
    console.log("Image:", file.name, file.type, file.size);
    console.log("Reference attached:", Boolean(refInput.files[0]));

    let response;

    try {
      response = await postImage(file, controller.signal);
    } catch (error) {
      if (error.name === "AbortError") throw error;

      console.warn("First request failed; retrying once.", error);
      showStatus("Backend connection failed. Retrying in 5 seconds...");

      await new Promise((resolve) => setTimeout(resolve, 5000));

      response = await postImage(file, controller.signal);
    }

    const responseText = await response.text();
    let result;

    try {
      result = JSON.parse(responseText);
    } catch {
      throw new Error(
        `Backend returned invalid JSON (HTTP ${response.status}).`
      );
    }

    if (!response.ok) {
      throw new Error(
        result.error ||
        result.message ||
        `Backend returned HTTP ${response.status}.`
      );
    }

    if (!result.image) {
      throw new Error("Backend response does not contain a restored image.");
    }

    await displayRestoredImage(result.image, canvasAfter);

    if (result.metrics) {
      updateMetrics(result.metrics);
    } else {
      console.warn("Backend did not return image metrics.");
    }

    createDownloadLink(result.image);
    showStatus("Image restoration completed successfully.");

    console.log("Restoration completed.");
  } catch (error) {
    console.error("Image restoration failed:", error);

    clearMetrics();

    if (error.name === "AbortError") {
      showStatus("Request timed out. Try a smaller image or retry.");
    } else if (error instanceof TypeError) {
      showStatus(
        "Could not connect to the backend. Check that Flask is running."
      );
    } else {
      showStatus(`Restoration failed: ${error.message}`);
    }
  } finally {
    clearTimeout(timeoutId);
    isProcessing = false;
  }
}

/* =========================================================
   DISPLAY RESTORED IMAGE
   ========================================================= */

function displayRestoredImage(imageData, targetCanvas) {
  return new Promise((resolve, reject) => {
    const restoredImage = new Image();

    restoredImage.onload = () => {
      targetCanvas.width = restoredImage.naturalWidth;
      targetCanvas.height = restoredImage.naturalHeight;

      const context = targetCanvas.getContext("2d");

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

    restoredImage.onerror = () => {
      reject(new Error("The backend returned an invalid image."));
    };

    restoredImage.src = imageData.startsWith("data:image")
      ? imageData
      : `data:image/png;base64,${imageData}`;
  });
}

/* =========================================================
   DOWNLOAD RESTORED IMAGE
   ========================================================= */

function createDownloadLink(imageData) {
  downloadBtn.href = imageData;
  downloadBtn.download = "deepsea-restored.png";
  downloadBtn.hidden = false;
}

/* =========================================================
   RESTORATION METRICS
   ========================================================= */

function updateMetrics(metrics) {
  console.log("Restoration metrics:", metrics);

  const configs = {
    psnr: { decimals: 2, suffix: " dB" },
    ssim: { decimals: 4, suffix: "" },
    uiqm: { decimals: 4, suffix: "" },
    uciqe: { decimals: 4, suffix: "" }
  };

  for (const [key, config] of Object.entries(configs)) {
    const row = document.querySelector(
      `.metric-row[data-metric="${key}"]`
    );

    if (!row) continue;

    const valueElement = row.querySelector(".metric-val");
    const fillElement = row.querySelector(".metric-fill");
    const rawValue = metrics[key];

    if (rawValue === undefined || rawValue === null) {
      valueElement.textContent =
        key === "psnr" || key === "ssim"
          ? "Needs reference"
          : "N/A";

      fillElement.style.width = "0%";
      continue;
    }

    const value = Number(rawValue);

    if (!Number.isFinite(value)) {
      valueElement.textContent = "N/A";
      fillElement.style.width = "0%";
      continue;
    }

    valueElement.textContent = value.toFixed(config.decimals) + config.suffix;

    let percentage;

    if (key === "psnr") {
      percentage = (value / 40) * 100;
    } else if (key === "ssim") {
      percentage = value * 100;
    } else if (key === "uiqm") {
      percentage = (value / UIQM_BAR_MAX) * 100;
    } else {
      percentage = (value / UCIQE_BAR_MAX) * 100;
    }

    percentage = Math.max(0, Math.min(100, percentage));
    fillElement.style.width = `${percentage}%`;
  }
}

/* =========================================================
   TRAINING HISTORY CHARTS
   ========================================================= */

const chartDefinitions = [
  { key: "psnr", canvasId: "psnrChart", label: "PSNR (dB)" },
  { key: "ssim", canvasId: "ssimChart", label: "SSIM" },
  { key: "uiqm", canvasId: "uiqmChart", label: "UIQM" },
  { key: "uciqe", canvasId: "uciqeChart", label: "UCIQE" }
];

function getMetricValue(record, key) {
  const aliases = {
    psnr: ["psnr", "PSNR", "val_psnr", "validation_psnr"],
    ssim: ["ssim", "SSIM", "val_ssim", "validation_ssim"],
    uiqm: ["uiqm", "UIQM", "val_uiqm", "validation_uiqm"],
    uciqe: ["uciqe", "UCIQE", "val_uciqe", "validation_uciqe"]
  };

  for (const name of aliases[key]) {
    if (record[name] !== undefined && record[name] !== null) {
      const value = Number(record[name]);

      if (Number.isFinite(value)) return value;
    }
  }

  return null;
}

function drawTrainingChart(definition, records) {
  const canvas = document.getElementById(definition.canvasId);

  if (!canvas) return 0;

  if (typeof Chart === "undefined") {
    throw new Error(
      "Chart.js did not load. Check your internet connection and reload."
    );
  }

  const points = records
    .map((record, index) => ({
      epoch: Number(record.epoch ?? record.epochs ?? index + 1),
      value: getMetricValue(record, definition.key)
    }))
    .filter(point =>
      Number.isFinite(point.epoch) && point.value !== null
    );

  if (trainingCharts[definition.key]) {
    trainingCharts[definition.key].destroy();
  }

  trainingCharts[definition.key] = new Chart(canvas, {
    type: "line",
    data: {
      labels: points.map(point => point.epoch),
      datasets: [{
        label: definition.label,
        data: points.map(point => point.value),
        borderColor: "#79c7ce",
        backgroundColor: "rgba(121,199,206,0.16)",
        borderWidth: 2,
        pointRadius: 5,
        pointHoverRadius: 7,
        fill: true,
        tension: 0.2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#e9f1f2" }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Epoch",
            color: "#9bb6bc"
          },
          ticks: { color: "#9bb6bc" },
          grid: { color: "rgba(180,220,225,0.10)" }
        },
        y: {
          title: {
            display: true,
            text: definition.label,
            color: "#9bb6bc"
          },
          ticks: { color: "#9bb6bc" },
          grid: { color: "rgba(180,220,225,0.10)" }
        }
      }
    }
  });

  return points.length;
}

async function loadTrainingHistory() {
  historyStatus.textContent = "Loading training history...";

  try {
    const response = await fetch(
      `${API_URL}/api/training-history?t=${Date.now()}`
    );

    if (!response.ok) {
      throw new Error(`History API returned HTTP ${response.status}`);
    }

    const data = await response.json();
    console.log("Training history received:", data);

    const records = Array.isArray(data)
      ? data
      : data.history ?? data.epochs ?? data.metrics ?? [];

    if (!Array.isArray(records) || records.length === 0) {
      throw new Error(
        "No epoch records returned. Check training_history.json."
      );
    }

    let totalPoints = 0;

    for (const definition of chartDefinitions) {
      totalPoints += drawTrainingChart(definition, records);
    }

    if (totalPoints === 0) {
      throw new Error(
        "No compatible metric values found in the history JSON."
      );
    }

    historyStatus.textContent =
      `Loaded ${records.length} training record(s); ` +
      `${totalPoints} metric point(s) plotted.`;
  } catch (error) {
    console.error("Training history loading failed:", error);

    historyStatus.textContent =
      `Could not load training graphs: ${error.message}`;
  }
}

refreshHistoryBtn.addEventListener("click", loadTrainingHistory);

loadTrainingHistory();
