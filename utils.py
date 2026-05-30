import pandas as pd

def predict_risk(model, feature_columns, df):

    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict probability of attrition
    df["Risk_Prob"] = model.predict_proba(df)[:, 1]

    return df