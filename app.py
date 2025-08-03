# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="Shopper Spectrum", layout="centered")

st.title("🛒 Shopper Spectrum: E-Commerce Intelligence App")

# Sidebar navigation
page = st.sidebar.radio("📌 Select Module", ["Home", "Customer Segmentation", "Product Recommendation"])

# Load models and data
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_files():
    scaler = joblib.load("rfm_scaler.pkl")
    kmeans = joblib.load("rfm_kmeans_model.pkl")
    rfm_data = pd.read_csv("rfm_clustered.csv")

    # Generate CustomerID vs StockCode matrix
    pivot = pd.pivot_table(
        rfm_data,
        values='InvoiceNo',
        index='CustomerID',
        columns='StockCode',
        aggfunc='count'
    ).fillna(0)

    # Compute cosine similarity matrix between products
    product_similarity = cosine_similarity(pivot.T)
    product_similarity_df = pd.DataFrame(product_similarity, index=pivot.columns, columns=pivot.columns)

    # Load product name mapping
    product_dict = pd.read_csv("product_description_map.csv", index_col=0).to_dict()["Description"]
    reverse_lookup = {v: k for k, v in product_dict.items()}

    return scaler, kmeans, rfm_data, product_similarity_df, product_dict, reverse_lookup

# ------------------------ MODULE: Home ------------------------
if page == "Home":
    st.subheader("📈 Project Overview")
    st.markdown("""
    Welcome to **Shopper Spectrum**!  
    This app provides two smart features:
    
    1. 🎯 **Customer Segmentation** using RFM analysis (Recency, Frequency, Monetary)
    2. 🛍️ **Product Recommendation** using item-based collaborative filtering

    Built using **Streamlit + Scikit-learn**.
    """)

# ------------------------ MODULE: Customer Segmentation ------------------------
elif page == "Customer Segmentation":
    st.subheader("📊 Customer Segmentation")

    with st.form("segmentation_form"):
        recency = st.number_input("📅 Recency (days since last purchase)", min_value=0, step=1)
        frequency = st.number_input("🔁 Frequency (number of purchases)", min_value=0, step=1)
        monetary = st.number_input("💰 Monetary (total amount spent)", min_value=0.0, step=1.0)

        submit = st.form_submit_button("Predict Segment")

    if submit:
        input_scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_scaled)[0]

        label_map = {
            0: "🟡 Regular Shopper",
            1: "🔴 At-Risk Shopper",
            2: "🟢 High-Value Shopper",
            3: "🔵 Occasional Shopper"
        }

        segment = label_map.get(cluster, "Unknown")
        st.success(f"Predicted Segment: **{segment}**")

# ------------------------ MODULE: Product Recommendation ------------------------
elif page == "Product Recommendation":
    st.subheader("🎯 Product Recommender")

    product_input = st.text_input("🔍 Enter a Product Name (e.g., GREEN VINTAGE SPOT BEAKER)")

    if st.button("Get Recommendations"):
        if product_input.strip().upper() in reverse_lookup:
            stockcode = reverse_lookup[product_input.strip().upper()]
            similar = product_similarity_df[stockcode].sort_values(ascending=False)[1:6]

            st.markdown("### 🛒 Top 5 Similar Products")
            for code in similar.index:
                name = product_dict.get(code, "Unknown")
                st.markdown(f"- **{name}** (Code: `{code}`) — Similarity: `{round(similar[code], 3)}`")
        else:
            st.error("❌ Product not found. Please enter an exact product name.")

