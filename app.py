import streamlit as st
import pandas as pd
import plotly.express as px

# ================= PAGE CONFIG (MUST BE FIRST) =================
st.set_page_config(page_title="HR AI Dashboard", layout="wide")

# ================= HEADER =================
st.markdown("""
## 💼 HR Attrition AI System
This system predicts employee attrition risk using Machine Learning and provides:
- Risk scoring
- Employee segmentation
- HR action recommendations
- Explainable insights for decision-making
""")

# ================= LOAD DATA =================
df = pd.read_csv("outputs/predictions.csv")

st.title("💼 HR Attrition AI Intelligence System")

# ================= KPI =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Employees", len(df))
col2.metric("High Risk", len(df[df["Risk_Level"] == "High"]))
col3.metric("Avg Risk Score", round(df["Risk_Score"].mean(), 2))
col4.metric("Attrition Rate", round(df["Attrition"].mean(), 2))

st.markdown("---")

# ================= RISK DISTRIBUTION =================
st.subheader("📊 Risk Distribution")

fig = px.pie(df, names="Risk_Level")
st.plotly_chart(fig, use_container_width=True)

# ================= OVERTIME IMPACT =================
st.subheader("⏱ OverTime Impact")

overtime_df = df.groupby("OverTime")["Attrition"].mean().reset_index()

fig = px.bar(overtime_df, x="OverTime", y="Attrition")
st.plotly_chart(fig, use_container_width=True)

# ================= PROMOTION IMPACT =================
st.subheader("📈 Promotion Delay Impact")

promo_df = df.groupby("YearsSinceLastPromotion")["Attrition"].mean().reset_index()

fig = px.line(promo_df,
              x="YearsSinceLastPromotion",
              y="Attrition")

st.plotly_chart(fig, use_container_width=True)

# ================= INCOME IMPACT (FIXED SAFE VERSION) =================
st.subheader("💰 Income Impact")

df_income = df.copy()

# Safe binning (prevents crash)
df_income = df_income.dropna(subset=["MonthlyIncome"])

df_income["IncomeBand"] = pd.qcut(
    df_income["MonthlyIncome"],
    5,
    duplicates="drop"
).astype(str)

income_df = df_income.groupby("IncomeBand")["Attrition"].mean().reset_index()

fig = px.bar(
    income_df,
    x="IncomeBand",
    y="Attrition",
    color="Attrition",
    title="Income vs Attrition Risk"
)

st.plotly_chart(fig, use_container_width=True)