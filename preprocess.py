import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    df = df.copy()

    # Convert target
    if df['Attrition'].dtype == 'object':
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Drop duplicates (important in real projects)
    df = df.drop_duplicates()

    # Handle missing values (simple strategy)
    df = df.fillna(df.median(numeric_only=True))

    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df