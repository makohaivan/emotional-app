// script.js
// - Captures a frame from the webcam
// - Sends a base64 image (dataURL) to /predict as JSON
// - Displays the returned emotion and confidence
// - Shows history in browser console when requested

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snapBtn = document.getElementById('snap');
const historyBtn = document.getElementById('historyBtn');
const resultEl = document.getElementById('result');

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error('Camera error:', err);
    resultEl.textContent = 'Unable to access camera. Check permissions.';
  }
}

// Convert canvas to data URL and POST to /predict
async function sendFrameForPrediction() {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/png');

  resultEl.textContent = 'Sending image...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });

    if (!res.ok) {
      const txt = await res.text();
      console.error('Server returned error:', txt);
      resultEl.textContent = 'Server error. See console.';
      return;
    }

    const json = await res.json();

    if (json.emotion) {
      const confPct = (json.confidence * 100).toFixed(1);
      resultEl.textContent = `Emotion: ${json.emotion} (${confPct}%)`;
    } else if (json.message) {
      resultEl.textContent = json.message;
    } else {
      resultEl.textContent = 'No face detected.';
    }
  } catch (err) {
    console.error('Prediction error:', err);
    resultEl.textContent = 'Network error. See console.';
  }
}

// Fetch history and log to console
async function fetchHistory() {
  try {
    const res = await fetch('/history');
    if (!res.ok) {
      resultEl.textContent = 'Could not fetch history.';
      return;
    }
    const json = await res.json();
    console.log('Prediction history (most recent first):', json.history);
    alert('History printed to the developer console. Open console to view.');
  } catch (err) {
    console.error('History fetch error:', err);
    alert('Could not fetch history. See console.');
  }
}

// Event listeners
snapBtn.addEventListener('click', sendFrameForPrediction);
historyBtn.addEventListener('click', fetchHistory);

// Start automatically
startCamera();
