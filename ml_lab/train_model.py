import pandas as pd
import numpy as np
import json
import os

# Load Data
try:
    df = pd.read_csv('synthetic_data.csv')
except FileNotFoundError:
    print("Error: synthetic_data.csv not found. Run generate_data.py first.")
    exit(1)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features & Labels
X = df[['duration', 'scroll_depth', 'key_count', 'switch_count']].values
y = df['label'].values

# Split Train/Test (80/20)
split_idx = int(0.8 * len(df))
X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Normalize (Fit on Train, Apply to Test)
min_vals = X_train_raw.min(axis=0)
max_vals = X_train_raw.max(axis=0)

def normalize(data, min_v, max_v):
    return (data - min_v) / (max_v - min_v + 1e-8)

X_train = normalize(X_train_raw, min_vals, max_vals)
X_test = normalize(X_test_raw, min_vals, max_vals)

# Logistic Regression with L2 Regularization
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, learning_rate=0.01, epochs=2000, lambda_reg=0.1):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    for epoch in range(epochs):
        # Forward
        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)
        
        # Gradient with L2 Regularization term for weights
        dw = (1/m) * np.dot(X.T, (predictions - y)) + (lambda_reg/m) * weights
        db = (1/m) * np.sum(predictions - y)
        
        # Update
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if epoch % 200 == 0:
            loss = -np.mean(y * np.log(predictions + 1e-8) + (1-y) * np.log(1-predictions + 1e-8))
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    return weights, bias

print("Training Logistic Regression with L2 Regularization...")
weights, bias = train_logistic_regression(X_train, y_train, lambda_reg=0.1)

# Evaluate on Test Set
z_test = np.dot(X_test, weights) + bias
preds_test = (sigmoid(z_test) > 0.5).astype(int)
accuracy = np.mean(preds_test == y_test)
print(f"Test Set Accuracy: {accuracy*100:.2f}%")

# Export Weights
# Path relative to this script (ml_lab/train_model.py) -> ../extension/model
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, '../extension/model')
if not os.path.exists(output_path):
    os.makedirs(output_path)

model_data = {
    "weights": weights.tolist(),
    "bias": bias,
    "min_vals": min_vals.tolist(),
    "max_vals": max_vals.tolist()
}

with open(os.path.join(output_path, 'weights.json'), 'w') as f:
    json.dump(model_data, f)

print(f"Saved weights to {output_path}/weights.json")

# Also export predictions for the full dataset so you can inspect what the
# model predicted for each synthetic example (probability + predicted label).
X_all_raw = X
X_all = normalize(X_all_raw, min_vals, max_vals)

z_all = np.dot(X_all, weights) + bias
probs_all = sigmoid(z_all)
preds_all = (probs_all > 0.5).astype(int)

pred_df = df.copy()
pred_df['pred'] = preds_all
pred_df['prob'] = probs_all

predictions_path = os.path.join(output_path, 'predictions.csv')
pred_df.to_csv(predictions_path, index=False)
print(f"Saved predictions to {predictions_path}")
