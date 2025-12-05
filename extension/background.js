importScripts('lib/xgb_compiled.js', 'lib/inference.js');

const model = new XGBFocusModel();
let isModelLoaded = false;

// Initialize model
(async () => {
    await model.load();
    isModelLoaded = true;
    console.log("Focus Tracker: Background model loaded.");
})();

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'HEARTBEAT') {
        handleHeartbeat(message.payload, sender).then(sendResponse);
        return true; // Keep channel open for async response
    }
});

async function handleHeartbeat(payload, sender) {
    if (!isModelLoaded) {
        return { status: 'LOADING', reason: 'Model initializing...' };
    }

    const { duration, scrollDepth, keyCount } = payload;
    
    // We don't have tab switching count from content script easily, 
    // but we could track it in background if needed. 
    // For now, let's assume 0 or try to track it separately.
    // The model expects: predict(duration, scrollDepth, keyCount, switchCount)
    const switchCount = 0; // Placeholder

    const prediction = model.predict(duration, scrollDepth, keyCount, switchCount);
    
    const status = prediction.label === 1 ? 'FOCUSED' : 'DISTRACTED';
    const reason = prediction.reason;

    // Update Current Session in Storage
    const sessionData = {
        timestamp: Date.now(),
        duration: duration,
        label: prediction.label,
        reason: reason,
        url: sender.tab ? sender.tab.url : 'unknown'
    };

    // We only update 'currentSession' for the active tab. 
    // Ideally we should track sessions per tab, but for simplicity:
    await chrome.storage.local.set({ currentSession: sessionData });

    // Also append to history if it's a "significant" update or end of session?
    // For now, popup handles history by reading 'sessions'. 
    // We should probably push to 'sessions' only when a tab closes or URL changes.
    // But the user just wants the "Tracking..." message fixed.
    
    return { status, reason };
}

// Optional: Listen for tab updates to track switches (if we want to implement that feature later)
chrome.tabs.onActivated.addListener(() => {
    // Increment switch count logic here if needed
});
