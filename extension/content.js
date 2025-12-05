// State
let scrollDepth = 0;
let keyCount = 0;
let startTime = Date.now();
let timerInterval = null;
let elapsedSeconds = 0; // accumulated while visible
let lastStart = Date.now();
let isVisible = !document.hidden;

// Overlay Elements
let overlayRoot = null;
let timerEl = null;
let statusEl = null;

// --- Tracking ---

// Scroll Tracking (Debounced)
let scrollTimeout;
window.addEventListener('scroll', () => {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(() => {
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollTop = window.scrollY;
    if (docHeight > 0) {
      const currentDepth = (scrollTop / docHeight) * 100;
      if (currentDepth > scrollDepth) {
        scrollDepth = currentDepth;
      }
    }
  }, 100);
});

// Key Tracking
window.addEventListener('keydown', () => {
  keyCount++;
});

// --- Overlay ---

function createOverlay() {
  const host = document.createElement('div');
  host.style.position = 'fixed';
  host.style.bottom = '20px';
  host.style.right = '20px';
  host.style.zIndex = '999999';
  document.body.appendChild(host);

  const shadow = host.attachShadow({ mode: 'open' });
  
  const style = document.createElement('style');
  style.textContent = `
    .container {
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 10px 15px;
      border-radius: 8px;
      font-family: sans-serif;
      font-size: 14px;
      display: flex;
      flex-direction: column;
      align-items: center;
      box-shadow: 0 4px 6px rgba(0,0,0,0.1);
      transition: background 0.3s;
    }
    .timer {
      font-weight: bold;
      font-size: 18px;
    }
    .status {
      font-size: 10px;
      text-transform: uppercase;
      margin-top: 4px;
      opacity: 0.8;
    }
    .reason {
      font-size: 11px;
      color: #fff;
      margin-top: 4px;
      font-style: italic;
    }
    .focused { border-left: 4px solid #4caf50; }
    .distracted { border-left: 4px solid #f44336; }
  `;
  
  const container = document.createElement('div');
  container.className = 'container';
  
  timerEl = document.createElement('div');
  timerEl.className = 'timer';
  timerEl.textContent = '00:00';
  
  statusEl = document.createElement('div');
  statusEl.className = 'status';
  statusEl.textContent = 'Tracking...';

  const reasonEl = document.createElement('div');
  reasonEl.className = 'reason';
  reasonEl.id = 'reason';
  reasonEl.textContent = 'Analyzing...';
  
  container.appendChild(timerEl);
  container.appendChild(statusEl);
  container.appendChild(reasonEl);
  shadow.appendChild(style);
  shadow.appendChild(container);
  
  overlayRoot = container;
}

function updateTimer() {
  const now = Date.now();
  let diffSec = elapsedSeconds;
  if (isVisible && lastStart) {
    diffSec += Math.floor((now - lastStart) / 1000);
  }
  const mins = Math.floor(diffSec / 60).toString().padStart(2, '0');
  const secs = (diffSec % 60).toString().padStart(2, '0');
  if (timerEl) timerEl.textContent = `${mins}:${secs}`;
}

// --- Communication ---

// Send metrics to background every 5 seconds
function startHeartbeat() {
  if (timerInterval) return;
  lastStart = Date.now();
  isVisible = true;
  timerInterval = setInterval(() => {
    const duration = elapsedSeconds + ((Date.now() - lastStart) / 1000);
  
  try {
    if (!chrome.runtime?.id) {
      clearInterval(timerInterval);
      return;
    }

    chrome.runtime.sendMessage({
      type: 'HEARTBEAT',
      payload: {
        duration,
        scrollDepth,
        keyCount
      }
    }, (response) => {
      if (chrome.runtime.lastError) {
        console.log("Focus Tracker: Connection lost. Refresh page.");
        clearInterval(timerInterval);
        return;
      }

      if (response && response.status) {
        if (statusEl) statusEl.textContent = response.status;
        
        const reasonDiv = overlayRoot.querySelector('#reason');
        if (reasonDiv && response.reason) reasonDiv.textContent = response.reason;

        if (overlayRoot) {
          if (response.status === 'FOCUSED') {
            overlayRoot.style.borderLeft = '4px solid #4caf50';
          } else {
            overlayRoot.style.borderLeft = '4px solid #f44336';
          }
        }
      }
    });
  } catch (e) {
    console.log("Focus Tracker: Context invalidated.");
    clearInterval(timerInterval);
  }
  }, 5000);
}

function stopHeartbeat() {
  if (!timerInterval) return;
  clearInterval(timerInterval);
  timerInterval = null;
  // accumulate elapsed
  elapsedSeconds += Math.floor((Date.now() - lastStart) / 1000);
  isVisible = false;
}

// Initialize
function init() {
  if (document.body) {
    createOverlay();
    setInterval(updateTimer, 1000);
  } else {
    // Retry if body is still null (rare but possible with frames)
    setTimeout(init, 100);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

// Pause heartbeat when tab hidden, resume when visible
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopHeartbeat();
  } else {
    startHeartbeat();
  }
});

// Start heartbeat initially only if visible
if (!document.hidden) startHeartbeat();
