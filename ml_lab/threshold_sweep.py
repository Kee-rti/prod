import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
except Exception:
    raise ImportError('xgboost is required; install with `pip install xgboost`')


def load_model(path):
    clf = XGBClassifier()
    clf.load_model(path)
    return clf


def load_data(path):
    df = pd.read_csv(path)
    X = df[['duration', 'scroll_depth', 'key_count', 'switch_count']].values
    y = df['label'].values
    return X, y, df


def sweep_thresholds(probs, y):
    thresholds = np.linspace(0, 1, 101)
    rows = []
    for t in thresholds:
        preds1 = (probs > t).astype(int)
        # class1 metrics
        prec1 = precision_score(y, preds1, zero_division=0)
        rec1 = recall_score(y, preds1, zero_division=0)
        f11 = f1_score(y, preds1, zero_division=0)

        # class0 metrics (treat prob0 = 1 - prob1)
        preds0 = (probs <= t).astype(int)  # 1 means class0 predicted
        y0 = (y == 0).astype(int)
        prec0 = precision_score(y0, preds0, zero_division=0)
        rec0 = recall_score(y0, preds0, zero_division=0)
        f10 = f1_score(y0, preds0, zero_division=0)

        rows.append({
            'threshold': float(t),
            'precision_1': float(prec1),
            'recall_1': float(rec1),
            'f1_1': float(f11),
            'precision_0': float(prec0),
            'recall_0': float(rec0),
            'f1_0': float(f10)
        })
    return rows


def choose_threshold(rows):
    # Prefer thresholds that give recall for class0 >= 0.99 and precision_1 >= 0.95
    candidates = [r for r in rows if r['recall_0'] >= 0.99 and r['precision_1'] >= 0.95]
    if not candidates:
        candidates = [r for r in rows if r['recall_0'] >= 0.99]
    if not candidates:
        # fallback: choose threshold that maximizes recall_0 while keeping precision_1 >= 0.9
        candidates = [r for r in rows if r['precision_1'] >= 0.9]
    if candidates:
        # choose one with max recall_0, tie-breaker max precision_1
        candidates.sort(key=lambda r: (r['recall_0'], r['precision_1']), reverse=True)
        return candidates[0]
    # final fallback: pick threshold with max recall_0
    rows.sort(key=lambda r: r['recall_0'], reverse=True)
    return rows[0]


def export_trees(clf, out_path):
    # Export per-tree JSON dumps
    booster = clf.get_booster()
    tree_strs = booster.get_dump(dump_format='json')
    # Save as array of strings (JSON encoded trees)
    with open(out_path, 'w') as f:
        json.dump(tree_strs, f)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, '../extension/model/xgb_model.json')
    data_path = os.path.join(root, 'synthetic_data.csv')
    out_dir = os.path.join(script_dir, '../extension/model')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print('XGBoost model not found. Run ml_lab/train_xgb.py first.')
        return

    clf = load_model(model_path)
    X, y, df = load_data(data_path)
    probs = clf.predict_proba(X)[:, 1]

    rows = sweep_thresholds(probs, y)
    with open(os.path.join(out_dir, 'threshold_sweep.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    chosen = choose_threshold(rows)
    with open(os.path.join(out_dir, 'threshold_report.txt'), 'w') as f:
        f.write(f"Chosen threshold: {chosen['threshold']}\n")
        f.write(json.dumps(chosen, indent=2))

    # Export trees for JS inference
    export_trees(clf, os.path.join(out_dir, 'xgb_trees.json'))

    print('Threshold sweep complete. Chosen threshold:', chosen['threshold'])


if __name__ == '__main__':
    main()
