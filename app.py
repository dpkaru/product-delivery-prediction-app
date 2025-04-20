import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('delivery_time_model.pkl', 'rb'))

# Feature names from training
feature_names = model.feature_names_in_

# UI
st.title("Delivery Time Predictor")

# Input widgets
product_category = st.selectbox("Product Category", ['Electronics', 'Furniture', 'Clothing', 'Toys', 'Books'])
customer_location = st.selectbox("Customer Location", ['New York', 'Los Angeles'])
shipping_method = st.selectbox("Shipping Method", ['Same-Day', 'Standard'])

# Prepare input
input_df = pd.DataFrame([[product_category, customer_location, shipping_method]],
                        columns=['Product_Category', 'Customer_Location', 'Shipping_Method'])

# One-hot encoding
input_encoded = pd.get_dummies(input_df).reindex(columns=feature_names, fill_value=0)

# Predict
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"📬 Estimated delivery time: **{round(prediction, 2)} days**")
