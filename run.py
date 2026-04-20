import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Insurance AI System", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Smart Insurance AI Dashboard")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"C:\Users\Thanuja C\OneDrive\Desktop\insurance_dataset.csv")


# ---------------- PREPROCESS ----------------
le = LabelEncoder()
for col in ["gender", "policy_type", "incident_type"]:
    df[col] = le.fit_transform(df[col])

# ---------------- FEATURE ENGINEERING ----------------
df["high_claim_flag"] = (df["claim_to_premium_ratio"] > 3).astype(int)
df["late_night_flag"] = (df["incident_hour"] > 20).astype(int)
df["no_witness_flag"] = (df["num_witnesses"] == 0).astype(int)
df["frequent_claimer"] = (df["previous_claims"] > 5).astype(int)

# ---------------- MODEL ----------------
X_reg = df.drop(["claim_amount", "fraud_label"], axis=1)
y_reg = df["claim_amount"]

reg_model = RandomForestRegressor(n_estimators=200)
reg_model.fit(X_reg, y_reg)

X_clf = df.drop(["fraud_label", "claim_amount"], axis=1)
y_clf = df["fraud_label"]

clf_model = RandomForestClassifier(n_estimators=300, class_weight="balanced")
clf_model.fit(X_clf, y_clf)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 Enter Claim Details")

age = st.sidebar.slider("Age", 18, 70)
income = st.sidebar.number_input("Annual Income", 200000, 2000000)
premium = st.sidebar.number_input("Premium Amount", 5000, 60000)
claims = st.sidebar.slider("Previous Claims", 0, 10)
hour = st.sidebar.slider("Incident Hour", 0, 23)
witness = st.sidebar.slider("Witnesses", 0, 5)
vehicle_age = st.sidebar.slider("Vehicle Age", 0, 15)
region_risk = st.sidebar.slider("Region Risk", 0.0, 1.0)

# Encoded defaults
gender = 0
policy_type = 1
incident_type = 0
police_report = 1

# Derived
claim_freq = claims / (age - 17)
claim_ratio = 3.0

high_claim_flag = int(claim_ratio > 3)
late_night_flag = int(hour > 20)
no_witness_flag = int(witness == 0)
frequent_claimer = int(claims > 5)

input_data = pd.DataFrame([[
    age, gender, income, policy_type, premium,
    incident_type, hour, witness, police_report,
    claims, vehicle_age, region_risk,
    claim_freq, claim_ratio,
    high_claim_flag, late_night_flag,
    no_witness_flag, frequent_claimer
]], columns=X_reg.columns)

# ---------------- PREDICTIONS ----------------
claim_pred = reg_model.predict(input_data)[0]
fraud_prob = clf_model.predict_proba(input_data)[0][1]

# ---------------- RISK ----------------
risk_score = (
    0.4 * fraud_prob +
    0.3 * (claim_ratio / 5) +
    0.2 * claim_freq +
    0.1 * (claims / 10)
)
risk_score = min(risk_score, 1)

# Risk color
if risk_score < 0.3:
    risk = "Low"
    color = "green"
elif risk_score < 0.7:
    risk = "Medium"
    color = "orange"
else:
    risk = "High"
    color = "red"

# ---------------- OUTPUT ----------------
st.subheader("📊 Prediction Results")

col1, col2, col3 = st.columns(3)

col1.markdown(f"""
<div class="card">
<h4>💰 Claim Amount</h4>
<h2>₹{int(claim_pred)}</h2>
</div>
""", unsafe_allow_html=True)

col2.markdown(f"""
<div class="card">
<h4>🚨 Fraud Probability</h4>
<h2>{round(fraud_prob,2)}</h2>
</div>
""", unsafe_allow_html=True)

col3.markdown(f"""
<div class="card">
<h4>⚠️ Risk Level</h4>
<h2 style="color:{color};">{risk}</h2>
</div>
""", unsafe_allow_html=True)

# ---------------- RISK BAR ----------------
st.subheader("🎯 Risk Score")
st.progress(int(risk_score * 100))

# ---------------- CHART ----------------
st.subheader("📊 Claim Distribution")

fig, ax = plt.subplots()
ax.hist(df["claim_amount"], bins=30)
st.pyplot(fig)