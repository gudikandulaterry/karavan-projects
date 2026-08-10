import pandas as pd
import joblib
import sys

# 1) Load feature_matrix for this run (you can generate it earlier in the pipeline)
df = pd.read_csv("feature_matrix.csv")

# 2) Separate features (same as training)
X = df.drop("deployment_status", axis=1, errors="ignore")

# 3) Load trained model
model = joblib.load("deployment_predictor.pkl")

# 4) Predict probability of failure (risk)
proba = model.predict_proba(X)[0][0]  # class 0 = failed
risk_score = proba * 100

print(f"RISK_SCORE={risk_score}")

# 5) Exit code based on threshold (40%)
if risk_score >= 40:
    sys.exit(1)   # risky → fail job
else:
    sys.exit(0)   # safe → pass job
