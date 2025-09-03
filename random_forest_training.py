# this code basically use the data from the excel file we created by another code, this code is a dense random forest,
# that connects the engineered features and the base model probabilities to the actual labels, so as to cerate a consolidated model,
# that gave better probs for future window predictions

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.utils import class_weight

# TODO: 0.537101557	0.557005189	0.583172721	0.597553744	0.633135656

# Load the data
df = pd.read_excel("meta_model_training_data.xlsx")

# Step 1: Filter out ambiguous samples (Suggestion #3)
# For label_60, keep only strong moves (e.g., ±1% range), drop weak/sideways cases
# This assumes you have access to the future return % or can define label granularity more strictly
# If not already done, here we keep all rows but show where this logic would go:

# Optional: Define cleaner signal labels (commented if not available)
# df = df[df["return_60"].abs() > 0.01]  # Filter sideways movements if you have return_60

# Step 2: Define features and labels
features = [
    "prob_timmy", "prob_jason", "prob_fedra", "prob_larry",
    "timmy_up_down_ratio", "timmy_drift", "timmy_volatility", "timmy_RSI", "timmy_bbw_30",
    "jason_up_down_ratio", "jason_drift", "jason_volatility", "jason_RSI", "jason_bbw_30", "jason_bbw_100",
    "fedra_up_down_ratio", "fedra_drift", "fedra_volatility", "fedra_RSI", "fedra_macd_hist_norm", "fedra_bbw_30", "fedra_bbw_100", "fedra_bbw_200",
    "larry_up_down_ratio", "larry_drift", "larry_volatility", "larry_RSI", "larry_macd_hist_norm", "larry_bbw_30", "larry_bbw_100", "larry_bbw_200"
]

# change this for different models for different future label testign (label 10, 30, etc)
target = "label_10"
X = df[features]
y = df[target]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

weights = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y)
class_weights = {0: weights[0] * 1.5, 1: weights[1]}  # emphasize class 0

# Step 4: Train the random forest with class_weight="balanced" (Suggestion #1)
base_rf = RandomForestClassifier(
    n_estimators=5000,
    max_depth=6,
    random_state=42,
    class_weight=class_weights
)

# Step 5: Calibrate the classifier (Suggestion #2)
calibrated_rf = CalibratedClassifierCV(base_rf, cv=5)
calibrated_rf.fit(X_train, y_train)

joblib.dump(calibrated_rf, "Tiberius_Elan_10_buy.pkl")

# Step 6: Predict and evaluate
y_pred = calibrated_rf.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: View calibrated probabilities
# y_proba = calibrated_rf.predict_proba(X_test)
# print(y_proba[:5])