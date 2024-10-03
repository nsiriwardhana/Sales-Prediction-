import streamlit as st
import numpy as np
import joblib
from scipy.special import inv_boxcox

# Load the saved model and lambda value
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    fitted_lambda = 0.37327999892652647  # Replace with the actual lambda value used in your Box-Cox transformation
    return model, fitted_lambda

model, fitted_lambda = load_model()

# Define categorical mappings
outlet_type_mapping = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
outlet_size_mapping = {'Small': 0, 'Medium': 1, 'High': 2}
outlet_location_type_mapping = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
outlet_identifier_mapping = {'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4, 'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9}

# Define the function to make predictions
def predict_sales(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    transformed_prediction = model.predict(input_array)[0]
    original_prediction = inv_boxcox(transformed_prediction, fitted_lambda)
    return original_prediction

# Set page config
st.set_page_config(page_title="Big Mart Sales Prediction", page_icon="🛒", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        color: #1E88E5;
        font-weight: bold;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
    }
    .stSelectbox {
        background-color: #f0f0f0;
        border-radius: 5px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown('<p class="big-font">Big Mart Sales Prediction 🛒</p>', unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([2, 1])

# Form in the first column
with col1:
    st.markdown("### Enter Product and Outlet Details")
    with st.form("prediction_form"):
        Item_MRP = st.number_input('Item MRP (₹)', min_value=0.0, max_value=300.0, value=100.0, step=0.1, format="%.2f")
        
        col_a, col_b = st.columns(2)
        with col_a:
            Outlet_Type = st.selectbox('Outlet Type', ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
            Outlet_Size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
            Outlet_Location_Type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
        
        with col_b:
            Outlet_Identifier = st.selectbox('Outlet Identifier', ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
            Item_Visibility_Interpolate = st.slider('Item Visibility', min_value=0.0, max_value=0.3, value=0.05, step=0.01)
            Outlet_Age = st.number_input('Outlet Age (years)', min_value=0, max_value=50, value=10, step=1)
        
        submitted = st.form_submit_button("Predict Sales")

# Results in the second column
with col2:
    st.markdown("### Prediction Results")
    if submitted:
        # Convert categorical inputs to numerical values using mappings
        Outlet_Type_encoded = outlet_type_mapping[Outlet_Type]
        Outlet_Identifier_encoded = outlet_identifier_mapping[Outlet_Identifier]
        Outlet_Size_encoded = outlet_size_mapping[Outlet_Size]
        Outlet_Location_Type_encoded = outlet_location_type_mapping[Outlet_Location_Type]

        # Gather input data into a list with encoded categorical variables
        input_data = [
            Item_MRP,
            Outlet_Type_encoded,
            Outlet_Identifier_encoded,
            Outlet_Size_encoded,
            Item_Visibility_Interpolate,
            Outlet_Location_Type_encoded,
            Outlet_Age
        ]
        
        # Predict sales
        original_prediction = predict_sales(input_data)
        
        # Display the prediction with some animations
        st.markdown("#### Sales Prediction")
        st.markdown(f'<p class="result">₹ {original_prediction:.2f}</p>', unsafe_allow_html=True)
        
        # Show balloons
        st.balloons()
        
        # Additional information
        st.info("This prediction is based on historical data and may not account for recent market changes.")
        st.markdown("### Prediction Breakdown")
        st.write("Key factors influencing the prediction:")
        st.write(f"- Item MRP: ₹{Item_MRP:.2f}")
        st.write(f"- Outlet Type: {Outlet_Type}")
        st.write(f"- Outlet Size: {Outlet_Size}")
        st.write(f"- Location Type: {Outlet_Location_Type}")

# Add some space
st.markdown("<br>", unsafe_allow_html=True)

# Disclaimer
st.markdown("---")
st.markdown("*Disclaimer: This is a predictive model and actual sales may vary. Always use professional judgment in conjunction with these predictions.*")