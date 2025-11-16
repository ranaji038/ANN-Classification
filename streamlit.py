import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ---------------------------
# Load Model & Encoders
# ---------------------------
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction")
st.markdown(
    """
    Predict whether a customer is likely to churn based on demographic and financial details.
    Fill the details below 👇
    """
)

# ---- Input Form ----
st.header("🔎 Enter Customer Details")
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
        gender = st.selectbox("⚧ Gender", label_encoder_gender.classes_)
        age = st.slider("🎂 Age", 18, 92, 30)

    with col2:
        credit_score = st.number_input("💳 Credit Score", min_value=0, max_value=1000, value=600)
        tenure = st.slider("📅 Tenure (Years)", 0, 10, 3)
        num_of_products = st.slider("🛍 Number of Products", 1, 4, 1)

    with col3:
        balance = st.number_input("💰 Balance", min_value=0.0, format="%.2f")
        estimated_salary = st.number_input("🧾 Estimated Salary", min_value=0.0, format="%.2f")
        has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
        is_active_member = st.selectbox("🔥 Active Member", [0, 1])

st.markdown("---")

# ---------------------------
# Prepare Input for Model
# ---------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)


# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🚀 Predict Churn", use_container_width=True):

    prediction = model.predict(input_data_scaled)
    probability = float(prediction[0][0])

    st.subheader("📌 Prediction Results")
    st.metric("Churn Probability", f"{probability:.2f}")

    if probability > 0.5:
        st.error("⚠️ **The customer is likely to churn.**")
    else:
        st.success("✅ **The customer is not likely to churn.**")
