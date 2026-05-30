def feature_engineering(df):
    df = df.copy()

    # Income efficiency
    df['Income_to_Experience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)

    # Promotion delay risk
    df['Promotion_Delay'] = (df['YearsSinceLastPromotion'] > 3).astype(int)

    # Career stagnation
    df['Stagnation'] = (
        (df['YearsInCurrentRole'] > 4) &
        (df['YearsSinceLastPromotion'] > 3)
    ).astype(int)

    # Engagement score
    if all(col in df.columns for col in ['JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction']):
        df['Engagement_Score'] = (
            df['JobSatisfaction'] +
            df['EnvironmentSatisfaction'] +
            df['RelationshipSatisfaction']
        ) / 3

    return df