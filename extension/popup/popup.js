document.addEventListener('DOMContentLoaded', async () => {
  const { sessions = [], currentSession } = await chrome.storage.local.get(['sessions', 'currentSession']);
  
  let focusedTime = 0;
  let distractedTime = 0;
  
  // 1. Add Historical Data
  sessions.forEach(s => {
    if (s.label === 1) focusedTime += s.duration;
    else distractedTime += s.duration;
  });

  // 2. Add Live Data (if valid)
  // Only add if it's recent (e.g., within last minute) to avoid stale state on browser restart
  // For simplicity, we just add it.
  if (currentSession) {
      if (currentSession.label === 1) focusedTime += currentSession.duration;
      else distractedTime += currentSession.duration;
  }

  const total = focusedTime + distractedTime;
  const focusedPct = total > 0 ? (focusedTime / total) * 100 : 0;
  
  document.getElementById('focused-time').textContent = `${Math.round(focusedTime/60)}m`;
  document.getElementById('distracted-time').textContent = `${Math.round(distractedTime/60)}m`;
  
  // Draw Chart (Simple Canvas Pie)
  const ctx = document.getElementById('chart').getContext('2d');
  
  // Background circle
  ctx.beginPath();
  ctx.arc(140, 100, 80, 0, 2 * Math.PI);
  ctx.fillStyle = '#eee';
  ctx.fill();
  
  // Focused slice
  if (total > 0) {
      const endAngle = (focusedPct / 100) * 2 * Math.PI;
      ctx.beginPath();
      ctx.moveTo(140, 100);
      ctx.arc(140, 100, 80, 0, endAngle);
      ctx.fillStyle = '#4caf50';
      ctx.fill();
      
      // Distracted slice (remainder)
      ctx.beginPath();
      ctx.moveTo(140, 100);
      ctx.arc(140, 100, 80, endAngle, 2 * Math.PI);
      ctx.fillStyle = '#f44336';
      ctx.fill();
  }

  // Last Session Info
  // Show current session if active, otherwise last history
  const displaySession = currentSession || (sessions.length > 0 ? sessions[0] : null);
  
  if (displaySession) {
    const reasonEl = document.getElementById('session-reason');
    const status = displaySession.label === 1 ? 'Focused' : 'Distracted';
    const reason = displaySession.reason || 'Unknown';
    reasonEl.innerHTML = `<strong>${status}</strong><br>${reason}`;
    reasonEl.style.color = displaySession.label === 1 ? '#4caf50' : '#f44336';
  }

  // Reset Button
  document.getElementById('reset-btn').addEventListener('click', async () => {
    if (confirm('Clear all history?')) {
      await chrome.storage.local.set({ sessions: [] });
      window.location.reload();
    }
  });

  // Demo harness: load XGB model and wire up predict button
  try {
    const model = new XGBFocusModel();
    await model.load();
    const resultEl = document.getElementById('predict-result');
    resultEl.textContent = 'Model loaded';

    document.getElementById('predict-btn').addEventListener('click', () => {
      const d = Number(document.getElementById('inp-duration').value) || 0;
      const s = Number(document.getElementById('inp-scroll').value) || 0;
      const k = Number(document.getElementById('inp-keys').value) || 0;
      const sw = Number(document.getElementById('inp-switches').value) || 0;
      const out = model.predict(d, s, k, sw);
      resultEl.textContent = `Prob: ${out.score.toFixed(3)} — ${out.label === 1 ? 'FOCUSED' : 'DISTRACTED'} (${out.reason})`;
      resultEl.style.color = out.label === 1 ? '#2e7d32' : '#c62828';
    });
  } catch (e) {
    console.warn('Demo model load failed', e);
  }
});
