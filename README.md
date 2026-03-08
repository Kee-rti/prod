# Prod — Time and Productivity Analysis

A browser extension that uses machine learning to detect whether you're **focused or distracted** while browsing — and shows you a breakdown of how you actually spent your time.

> Built by Kirti & Shikha

---

## What it does

Most people track screen time on their phones. Prod brings that awareness to your laptop or PC — with actual intelligence behind it.

- 🧠 **Classifies your browsing** as Focused or Distracted in real time
- 💬 **Explains its decision** — not just a label, but a reason
- 📊 **Shows a pie chart** of your total focused vs distracted time
- 🔒 **Privacy-first** — all inference runs locally in the browser, no data sent to a server

---

## How it works

The ML pipeline (in `ml_lab/`) was built in 6 steps:

1. **Define scenarios** — identify behavioural signals for Focused vs Distracted states
2. **Generate synthetic data** — labelled dataset with key behavioural features (`synthetic_data.csv`)
3. **Train with XGBoost** — learns non-linear behavioural patterns to classify focus level
4. **Baseline with Logistic Regression** — used as a fallback; data proved non-linearly separable
5. **Evaluate** — XGBoost significantly outperformed the baseline
6. **Export** — model trees compiled from JSON → JavaScript for use inside the extension

**Model performance (XGBoost):**
| Metric | Score |
|--------|-------|
| Accuracy | 0.9857 |
| Precision | 0.9856 |
| Recall | 0.9904 |
| F1 | 0.9880 |

---

## Project Structure

```
prod/
├── extension/
│   ├── manifest.json     # Extension configuration
│   ├── background.js     # Communication hub, model inference & storage
│   └── content.js        # Collects and sends tab data
├── ml_lab/               # Model training scripts (Python)
├── synthetic_data.csv    # Training dataset
└── requirements.txt      # Python dependencies
```

---

## Getting Started

### 1. Train the model (optional — pre-trained model included)

```bash
pip install -r requirements.txt
cd ml_lab
python train.py
```

### 2. Load the extension in Chrome/Edge

1. Go to `chrome://extensions` (or `edge://extensions`)
2. Enable **Developer Mode** (toggle, top right)
3. Click **Load unpacked** → select the `extension/` folder
4. The Prod icon will appear in your toolbar 

---

## Requirements

- Python 3.8+ (for retraining only)
- Google Chrome or Microsoft Edge

---

## Future Scope

- [ ] Personalized model training per user
- [ ] Expand tracking beyond the browser to desktop apps
- [ ] Gamified focus streaks & achievement badges
- [ ] Leaderboards and competitive focus scores
- [ ] Goal-based XP system


