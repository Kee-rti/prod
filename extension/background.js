importScripts('lib/xgb_compiled.js', 'lib/inference.js');

const model = new XGBFocusModel();
let isModelLoaded = false;

// Browser-level telemetry state.
// This is intentionally separate from ML inference so the browser is the
// source of truth for tab switches rather than the content script.
const TAB_TELEMETRY_KEY = 'tabTelemetryState';
let tabTelemetryState = {
    activeTabByWindow: {},
    switchCountByWindow: {},
    lastActivationAt: null
};

let telemetryStateReady = initializeTabTelemetryState();
let telemetryWriteChain = Promise.resolve();

// Initialize model
(async () => {
    await model.load();
    isModelLoaded = true;
    console.log("Focus Tracker: Background model loaded.");
})();

// Register browser event listeners synchronously for MV3 service-worker reliability.
chrome.tabs.onActivated.addListener((activeInfo) => {
    telemetryStateReady.then(() => registerTabActivation(activeInfo));
});

chrome.tabs.onRemoved.addListener((tabId) => {
    telemetryStateReady.then(() => removeTabFromTelemetryState(tabId));
});

chrome.windows.onRemoved.addListener((windowId) => {
    telemetryStateReady.then(() => removeWindowFromTelemetryState(windowId));
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'HEARTBEAT') {
        handleHeartbeat(message.payload, sender).then(sendResponse);
        return true; // Keep channel open for async response
    }
});

async function initializeTabTelemetryState() {
    try {
        const stored = await chrome.storage.session.get(TAB_TELEMETRY_KEY);
        if (stored[TAB_TELEMETRY_KEY]) {
            tabTelemetryState = {
                ...tabTelemetryState,
                ...stored[TAB_TELEMETRY_KEY]
            };
        }

        // Seed only windows that are not already represented. For existing
        // windows, preserve the previous active tab so an onActivated event
        // can correctly detect a switch after worker suspension.
        const activeTabs = await chrome.tabs.query({ active: true });
        for (const tab of activeTabs) {
            if (tab.id == null || tab.windowId == null) continue;

            const windowKey = String(tab.windowId);
            if (tabTelemetryState.activeTabByWindow[windowKey] == null) {
                tabTelemetryState.activeTabByWindow[windowKey] = tab.id;
            }

            if (tabTelemetryState.switchCountByWindow[windowKey] == null) {
                tabTelemetryState.switchCountByWindow[windowKey] = 0;
            }
        }

        await persistTabTelemetryState();
    } catch (error) {
        console.warn("Focus Tracker: Failed to initialize tab telemetry state.", error);
    }
}

function registerTabActivation(activeInfo) {
    const windowKey = String(activeInfo.windowId);
    const previousTabId = tabTelemetryState.activeTabByWindow[windowKey];

    // Count a switch only when the active tab actually changes.
    // First activation after startup is not treated as a switch.
    if (previousTabId != null && previousTabId !== activeInfo.tabId) {
        tabTelemetryState.switchCountByWindow[windowKey] =
            (tabTelemetryState.switchCountByWindow[windowKey] || 0) + 1;
    }

    tabTelemetryState.activeTabByWindow[windowKey] = activeInfo.tabId;
    tabTelemetryState.switchCountByWindow[windowKey] =
        tabTelemetryState.switchCountByWindow[windowKey] || 0;
    tabTelemetryState.lastActivationAt = Date.now();

    void persistTabTelemetryState();
}

function removeTabFromTelemetryState(tabId) {
    let changed = false;

    for (const [windowKey, activeTabId] of Object.entries(tabTelemetryState.activeTabByWindow)) {
        if (activeTabId === tabId) {
            delete tabTelemetryState.activeTabByWindow[windowKey];
            changed = true;
        }
    }

    if (changed) {
        void persistTabTelemetryState();
    }
}

function getTabSwitchCount(windowId) {
    if (windowId == null) return 0;
    return tabTelemetryState.switchCountByWindow[String(windowId)] || 0;
}

function removeWindowFromTelemetryState(windowId) {
    const windowKey = String(windowId);
    const hadActiveTab = Object.prototype.hasOwnProperty.call(
        tabTelemetryState.activeTabByWindow,
        windowKey
    );
    const hadSwitchCount = Object.prototype.hasOwnProperty.call(
        tabTelemetryState.switchCountByWindow,
        windowKey
    );

    if (!hadActiveTab && !hadSwitchCount) return;

    delete tabTelemetryState.activeTabByWindow[windowKey];
    delete tabTelemetryState.switchCountByWindow[windowKey];
    void persistTabTelemetryState();
}

function persistTabTelemetryState() {
    // Serialize writes so rapid tab switches cannot overwrite a newer state
    // with an older asynchronous write.
    telemetryWriteChain = telemetryWriteChain
        .then(() => chrome.storage.session.set({
            [TAB_TELEMETRY_KEY]: tabTelemetryState
        }))
        .catch((error) => {
            console.warn("Focus Tracker: Failed to persist tab telemetry state.", error);
        });

    return telemetryWriteChain;
}

async function handleHeartbeat(payload, sender) {
    await telemetryStateReady;

    if (!isModelLoaded) {
        return { status: 'LOADING', reason: 'Model initializing...' };
    }

    const { duration, scrollDepth, keyCount } = payload;

    // Real switch telemetry now comes from browser events, not the content
    // script. We deliberately do not feed it into XGBoost yet because the
    // existing model was trained with a different switchCount definition.
    // Retraining with the new feature semantics will be a separate step.
    const tabSwitchCount = getTabSwitchCount(sender.tab?.windowId);
    const modelSwitchCount = 0;

    const prediction = model.predict(
        duration,
        scrollDepth,
        keyCount,
        modelSwitchCount
    );

    const status = prediction.label === 1 ? 'FOCUSED' : 'DISTRACTED';
    const reason = prediction.reason;

    // Update Current Session in Storage
    const sessionData = {
        timestamp: Date.now(),
        duration: duration,
        label: prediction.label,
        reason: reason,
        url: sender.tab ? sender.tab.url : 'unknown',
        tabId: sender.tab?.id ?? null,
        windowId: sender.tab?.windowId ?? null,
        tabSwitchCount: tabSwitchCount
    };

    // We still only expose the latest session here. The next telemetry step
    // will replace this with explicit per-tab session segments.
    await chrome.storage.local.set({ currentSession: sessionData });

    return {
        status,
        reason,
        tabSwitchCount
    };
}
