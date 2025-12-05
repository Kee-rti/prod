import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_fscore_support)


def load_model(model_path):
    with open(model_path, 'r') as f:
        return json.load(f)


def normalize(data, min_v, max_v):
    min_v = np.array(min_v)
    max_v = np.array(max_v)
    return (data - min_v) / (max_v - min_v + 1e-8)


def predict_proba(X, weights, bias):
    z = np.dot(X, weights) + bias
    probs = 1 / (1 + np.exp(-z))
    return probs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, '../extension/model/weights.json')
    data_path = os.path.join(root, 'synthetic_data.csv')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run `train_model.py` first.")
        return
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Run `generate_data.py` first.")
        return

    model = load_model(model_path)
    weights = np.array(model['weights'])
    bias = float(model['bias'])
    min_vals = np.array(model['min_vals'])
    max_vals = np.array(model['max_vals'])

    df = pd.read_csv(data_path)
    X_raw = df[['duration', 'scroll_depth', 'key_count', 'switch_count']].values
    y = df['label'].values

    X = normalize(X_raw, min_vals, max_vals)
    probs = predict_proba(X, weights, bias)
    preds = (probs > 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds).tolist()

    # ROC AUC (if both classes present)
    try:
        roc_auc = roc_auc_score(y, probs)
    except Exception:
        roc_auc = None

    # Per-class precision/recall/f1
    per_class = precision_recall_fscore_support(y, preds, zero_division=0)

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm,
        'roc_auc': None if roc_auc is None else float(roc_auc),
        'per_class': {
            'precision': per_class[0].tolist(),
            'recall': per_class[1].tolist(),
            'f1': per_class[2].tolist(),
            'support': per_class[3].tolist()
        }
    }

    out_dir = os.path.join(script_dir, '../extension/model')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Write a short report
    report_lines = []
    report_lines.append(f"Accuracy: {accuracy:.4f}")
    report_lines.append(f"Precision: {precision:.4f}")
    report_lines.append(f"Recall: {recall:.4f}")
    report_lines.append(f"F1: {f1:.4f}")
    report_lines.append(f"ROC AUC: {'' if roc_auc is None else f'{roc_auc:.4f}' }")
    report_lines.append('Confusion Matrix:')
    report_lines.append(str(cm))
    report_lines.append('Per-class (precision, recall, f1, support):')
    for i in range(len(per_class[0])):
        report_lines.append(f"  Class {i}: {per_class[0][i]:.4f}, {per_class[1][i]:.4f}, {per_class[2][i]:.4f}, support={int(per_class[3][i])}")

    report_path = os.path.join(out_dir, 'eval_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Saved evaluation results to {json_path} and {report_path}")


if __name__ == '__main__':
    main()
