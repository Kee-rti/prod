class FocusModel {
  constructor() {
    this.weights = null;
    this.bias = null;
    this.minVals = null;
    this.maxVals = null;
  }

  async load() {
    try {
      const url = chrome.runtime.getURL('model/weights.json');
      const response = await fetch(url);
      const data = await response.json();
      
      this.weights = data.weights;
      this.bias = data.bias;
      this.minVals = data.min_vals;
      this.maxVals = data.max_vals;
      console.log("Model loaded:", data);
    } catch (e) {
      console.error("Failed to load model:", e);
    }
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  predict(duration, scrollDepth, keyCount, switchCount) {
    if (!this.weights) return { score: 0.5, label: 0, reason: 'Loading...' };

    // Normalize
    const features = [duration, scrollDepth, keyCount, switchCount];
    const featureNames = ['Duration', 'Scroll Depth', 'Key Activity', 'Tab Switching'];
    
    const normFeatures = features.map((v, i) => {
      return (v - this.minVals[i]) / (this.maxVals[i] - this.minVals[i] + 1e-8);
    });

    // Calculate contributions
    let z = this.bias;
    let contributions = [];
    
    for (let i = 0; i < 4; i++) {
      const contrib = normFeatures[i] * this.weights[i];
      z += contrib;
      contributions.push({ name: featureNames[i], value: contrib });
    }

    const score = this.sigmoid(z);
    const label = score > 0.5 ? 1 : 0;
    
    // Find dominant reason
    let reason = '';
    if (label === 1) {
      // Focused: Find highest positive contributor
      // Sort by value descending
      contributions.sort((a, b) => b.value - a.value);
      
      let max = contributions[0];
      
      // Don't cite Duration as the reason if it's too short (< 30s)
      if (max.name === 'Duration' && duration < 30) {
          // Try next best
          if (contributions[1].value > 0) {
              max = contributions[1];
          } else {
              // If no other positive contributors, maybe it shouldn't be focused?
              // But if the bias is high enough to keep it focused, we say "Good Focus"
              reason = "Good Focus"; 
          }
      }
      
      if (!reason) reason = `Good ${max.name}`;
      
    } else {
      // Distracted: Find lowest negative contributor (or smallest positive if all positive but bias is negative)
      const min = contributions.reduce((prev, current) => (prev.value < current.value) ? prev : current);
      // If the biggest drag is scroll, say "High Scroll"
      // We need to interpret the weight sign. 
      // If weight is negative, high feature value = negative contribution -> "High Scroll"
      // If weight is positive, low feature value = low contribution -> "Low Duration"
      
      if (this.weights[features.indexOf(features[contributions.indexOf(min)])] < 0) {
         reason = `High ${min.name}`;
      } else {
         reason = `Low ${min.name}`;
      }
    }

    return { score, label, reason };
  }
}

// Export for use in background/content
globalThis.FocusModel = FocusModel;

class XGBFocusModel {
  constructor() {
    this.trees = null; // array of node maps
    this.threshold = 0.86; // default chosen threshold from sweep
  }

  async load() {
    try {
      const treesUrl = chrome.runtime.getURL('model/xgb_trees.json');
      const resp = await fetch(treesUrl);
      const treeStrs = await resp.json();
      // treeStrs is an array of JSON strings (one per tree)
      this.trees = treeStrs.map(s => JSON.parse(s));

      // Check if compiled predictor is already loaded (via script tag or importScripts)
      if (typeof globalThis.predictXGB === 'function') {
        this.compiled = true;
      } else {
        console.log('Compiled XGB predictor not found, falling back to interpreter.');
        this.compiled = false;
      }

      // Build node maps for fast traversal
      this.nodeMaps = this.trees.map(tree => {
        const map = new Map();
        const stack = [tree];
        while (stack.length) {
          const node = stack.pop();
          map.set(node.nodeid, node);
          if (node.children) {
            for (const c of node.children) stack.push(c);
          }
        }
        return map;
      });

      // Try to load threshold report if present
      try {
        const thrUrl = chrome.runtime.getURL('model/threshold_report.txt');
        const thrResp = await fetch(thrUrl);
        if (thrResp.ok) {
          const txt = await thrResp.text();
          const m = txt.match(/Chosen threshold:\s*([0-9.]+)/);
          if (m) this.threshold = parseFloat(m[1]);
        }
      } catch (e) {
        // ignore
      }

      console.log('XGBoost model loaded,', this.trees.length, 'trees, threshold=', this.threshold);
    } catch (e) {
      console.error('Failed to load XGBoost model:', e);
    }
  }

  evalTree(nodeMap, startNodeId, features) {
    let node = nodeMap.get(startNodeId);
    while (node) {
      if (Object.prototype.hasOwnProperty.call(node, 'leaf')) {
        return node.leaf;
      }
      // split field is usually 'split' like 'f0'
      const splitName = node.split || node.feature || '';
      const cond = node.split_condition ?? node.threshold ?? node.split_condition;
      const fIndex = parseInt(String(splitName).replace(/[^0-9]/g, ''), 10);
      const val = features[fIndex];
      let nextId;
      if (val === null || val === undefined || Number.isNaN(val)) {
        nextId = node.missing;
      } else if (val < cond) {
        nextId = node.yes;
      } else {
        nextId = node.no;
      }
      node = nodeMap.get(nextId);
    }
    return 0;
  }

  predict(duration, scrollDepth, keyCount, switchCount) {
    if (!this.trees || !this.nodeMaps) return { score: 0.5, label: 0, reason: 'Loading XGB...' };

    const features = [duration, scrollDepth, keyCount, switchCount];

    // If compiled predictor is available, use it for speed
    if (this.compiled && typeof globalThis.predictXGB === 'function') {
      const out = globalThis.predictXGB(features, this.threshold);
      return { score: out.prob, label: out.label, reason: 'xgb_compiled' };
    }

    // Fallback: interpret trees
    let score = 0;
    for (let i = 0; i < this.nodeMaps.length; i++) {
      const nodeMap = this.nodeMaps[i];
      score += this.evalTree(nodeMap, 0, features);
    }

    // XGBoost outputs raw score (log-odds for binary: logistic)
    const prob = 1 / (1 + Math.exp(-score));
    const label = prob > this.threshold ? 1 : 0;

    return { score: prob, label, reason: 'xgb' };
  }
}

globalThis.XGBFocusModel = XGBFocusModel;
