const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusEl = document.getElementById('status');
const phraseEl = document.getElementById('phrase');
const apiUrlEl = document.getElementById('apiUrl');
const startBtn = document.getElementById('startBtn');
const switchBtn = document.getElementById('switchBtn');
const stopBtn = document.getElementById('stopBtn');

let stream = null;
let timer = null;
let lastSpoken = '';
let lastSpokenAt = 0;
let currentFacingMode = 'environment';

const SEND_INTERVAL_MS = 220;
const SPEAK_COOLDOWN_MS = 1500;

function setStatus(text) {
  statusEl.textContent = text;
}

function speakId(text) {
  const now = Date.now();
  if (!text) return;
  if (text === lastSpoken && now - lastSpokenAt < SPEAK_COOLDOWN_MS) return;

  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = 'id-ID';
  speechSynthesis.cancel();
  speechSynthesis.speak(utter);

  lastSpoken = text;
  lastSpokenAt = now;
}

async function sendFrame() {
  if (!stream) return;

  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;

  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

  try {
    const res = await fetch(apiUrlEl.value.trim(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_base64: dataUrl,
        max_hands: 2
      })
    });

    if (!res.ok) {
      setStatus(`error ${res.status}`);
      return;
    }

    const json = await res.json();
    const phrase = (json.phrase || '').trim();
    phraseEl.textContent = phrase || '-';
    setStatus('running');
    speakId(phrase);
  } catch (err) {
    setStatus(`request failed: ${err.message}`);
  }
}

async function startCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus('camera API tidak tersedia. Buka lewat HTTPS dan browser utama (bukan in-app browser).');
      return;
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: currentFacingMode },
      audio: false
    });
    video.srcObject = stream;
    await video.play();
    setStatus(`camera on (${currentFacingMode})`);

    if (timer) clearInterval(timer);
    timer = setInterval(sendFrame, SEND_INTERVAL_MS);
  } catch (err) {
    setStatus(`camera error: ${err.message}`);
  }
}

async function switchCamera() {
  currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
  stopCamera();
  await startCamera();
}

function stopCamera() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  setStatus('stopped');
}

startBtn.addEventListener('click', startCamera);
switchBtn.addEventListener('click', switchCamera);
stopBtn.addEventListener('click', stopCamera);
