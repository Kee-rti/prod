import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             make_scorer)
from sklearn.utils import resample

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost is required. Install with `pip install xgboost`.")


def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df[['duration', 'scroll_depth', 'key_count', 'switch_count']].values
    y = df['label'].values
    return X, y, df


def resample_balance(X, y, df, random_state=42):
    # Oversample minority class to balance classes
    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]
    if len(df_pos) == len(df_neg):
        return X, y, df
    if len(df_pos) > len(df_neg):
        df_min = df_neg
        df_maj = df_pos
    else:
        df_min = df_pos
        df_maj = df_neg

    df_min_upsampled = resample(df_min,
                                replace=True,
                                n_samples=len(df_maj),
                                random_state=random_state)
    df_bal = pd.concat([df_maj, df_min_upsampled])
    df_bal = df_bal.sample(frac=1, random_state=random_state).reset_index(drop=True)
    Xb = df_bal[['duration', 'scroll_depth', 'key_count', 'switch_count']].values
    yb = df_bal['label'].values
    return Xb, yb, df_bal


def compute_sample_weights(y):
    # Inverse frequency weighting to give more weight to minority class
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weight = np.array([weights[int(label)] for label in y], dtype=float)
    return sample_weight


def run_grid_search(X, y, sample_weight, cv_splits=5, random_state=42):
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.01]
    }

    # Optimize recall for class 0 (distracted) to reduce false negatives for that class
    scorer = make_scorer(recall_score, pos_label=0)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    gs = GridSearchCV(clf, param_grid, scoring=scorer, cv=cv, n_jobs=1, verbose=1)
    gs.fit(X, y, sample_weight=sample_weight)
    return gs


def evaluate_model(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds).tolist()
    try:
        roc = roc_auc_score(y, probs)
    except Exception:
        roc = None

    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm,
        'roc_auc': None if roc is None else float(roc)
    }, preds, probs


def save_model_and_results(model, out_dir, results, preds, probs, df):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'xgb_model.json')
    model.get_booster().save_model(model_path)

    with open(os.path.join(out_dir, 'eval_xgb_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    report_lines = []
    report_lines.append(f"Accuracy: {results['accuracy']:.4f}")
    report_lines.append(f"Precision: {results['precision']:.4f}")
    report_lines.append(f"Recall: {results['recall']:.4f}")
    report_lines.append(f"F1: {results['f1']:.4f}")
    report_lines.append(f"ROC AUC: {'' if results['roc_auc'] is None else f'{results['roc_auc']:.4f}' }")
    report_lines.append('Confusion Matrix:')
    report_lines.append(str(results['confusion_matrix']))

    with open(os.path.join(out_dir, 'eval_xgb_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))

    # Save predictions
    pred_df = df.copy()
    pred_df['pred_xgb'] = preds
    pred_df['prob_xgb'] = probs
    pred_df.to_csv(os.path.join(out_dir, 'predictions_xgb.csv'), index=False)

    print(f"Saved XGBoost model to {model_path} and evaluation to {out_dir}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    data_path = os.path.join(root, 'synthetic_data.csv')
    out_dir = os.path.join(script_dir, '../extension/model')

    if not os.path.exists(data_path):
        print("Data not found. Run `generate_data.py` first.")
        return

    X, y, df = load_data(data_path)

    # Option: oversample minority to balance
    X_bal, y_bal, df_bal = resample_balance(X, y, df)

    # Compute sample weights (inverse freq) on balanced data
    sample_weight = compute_sample_weights(y_bal)

    # Grid search cross-validation optimizing recall for class 0
    print("Starting GridSearchCV optimizing recall for class 0 (distracted) ...")
    gs = run_grid_search(X_bal, y_bal, sample_weight)

    print(f"Best params: {gs.best_params_}")

    # Refit best estimator on the balanced data with computed sample weights
    best = gs.best_estimator_
    best.fit(X_bal, y_bal, sample_weight=sample_weight)

    # Evaluate on original dataset (not balanced) to see real-world performance
    results, preds, probs = evaluate_model(best, X, y)

    save_model_and_results(best, out_dir, results, preds, probs, df)


if __name__ == '__main__':
    main()
