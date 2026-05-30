import pandas as pd
import os
from model import train_model

# Load data
df = pd.read_csv("dataset.csv")

print("Original Shape:", df.shape)

# Train model
model, X_test, y_test, features = train_model(df)

# Prepare full dataset
X_all = pd.get_dummies(df.drop("Attrition", axis=1)).reindex(columns=features, fill_value=0)

# Risk score
df["Risk_Score"] = model.predict_proba(X_all)[:, 1]

# Risk level
df["Risk_Level"] = pd.cut(
    df["Risk_Score"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low", "Medium", "High"]
)

# HR Action system
def action(x):
    if x > 0.7:
        return "Immediate Action"
    elif x > 0.4:
        return "Monitor"
    else:
        return "Safe"

df["HR_Action"] = df["Risk_Score"].apply(action)

# Feature importance
feat = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# SAVE FILES (IMPORTANT)
df.to_csv("outputs/predictions.csv", index=False)
feat.to_csv("outputs/feature_importance.csv", index=False)

print("✅ Files saved in outputs/")