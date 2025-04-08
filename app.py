import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset from CSV
df = pd.read_csv(order_details_dataset)


df = load_data(uploaded_file)    
st.write("Preview of your dataset:", df.head())

# User input
product = st.selectbox("Select Product Category", product_categories)
location = st.selectbox("Select Customer Location", locations)
shipping = st.selectbox("Select Shipping Method", shipping_methods)

if st.button("Predict Delivery Time"):
        user_input = pd.DataFrame([[product, location, shipping]],
        columns=['Product_Category', 'Customer_Location', 'Shipping_Method'])
        prediction = model.predict(user_input)[0]
        st.success(f" Estimated Delivery Time: **{prediction:.1f} days**")
else:
    st.info("Please upload a CSV file to get started.")
