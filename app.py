import streamlit as st
import numpy as np
import joblib
from scipy.special import inv_boxcox

# Load the saved model and lambda value
model = joblib.load('model.pkl')
fitted_lambda = 0.37327999892652647  # Replace with the actual lambda value used in your Box-Cox transformation

# Define categorical mappings
outlet_type_mapping = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
outlet_size_mapping = {'Small': 0, 'Medium': 1, 'High': 2}
outlet_location_type_mapping = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
outlet_identifier_mapping = {'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4, 'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9}

# Define the function to make predictions
def predict_sales(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    transformed_prediction = model.predict(input_array)[0]
    
    # Apply the inverse Box-Cox transformation to get the original sales prediction
    original_prediction = inv_boxcox(transformed_prediction, fitted_lambda)
    return original_prediction

# Set up the Streamlit app
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url("https://static.vecteezy.com/system/resources/thumbnails/021/935/373/small_2x/supermarket-aisle-perspective-view-free-vector.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    .title {
        color: white;
        font-size: 36px;
        text-align: center;
    }

    .input-label {
        color: white;
        font-weight: bold;
        display: block;
        margin-top: 20px;
        font-size: 22px;
    }

    .prediction-output {
        color: #FFD700;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
    }

    .feedback-message {
        color: white;
        font-size: 18px;
        text-align: center;
        margin-top: 20px;
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        font-size: 20px;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }

    .stButton > button:active {
        background-color: #3e8e41;
        color: white; /* Ensure font color remains white on click */
    }

    </style>
    """, unsafe_allow_html=True)

# Title with custom class for styling
st.markdown('<h1 class="title">Big Mart Sales Prediction</h1>', unsafe_allow_html=True)

# Create a form for user input
with st.form("prediction_form"):
    # Custom markdown labels for the selected features
    st.markdown('<span class="input-label">Item MRP</span>', unsafe_allow_html=True)
    Item_MRP = st.number_input('', min_value=0.0, max_value=300.0, value=100.0)

    st.markdown('<span class="input-label">Outlet Type</span>', unsafe_allow_html=True)
    Outlet_Type = st.selectbox('', ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

    st.markdown('<span class="input-label">Outlet Identifier</span>', unsafe_allow_html=True)
    Outlet_Identifier = st.selectbox('', ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])

    st.markdown('<span class="input-label">Outlet Size</span>', unsafe_allow_html=True)
    Outlet_Size = st.selectbox('', ['Small', 'Medium', 'High'])

    st.markdown('<span class="input-label">Item Visibility Interpolate</span>', unsafe_allow_html=True)
    Item_Visibility_Interpolate = st.number_input('', min_value=0.0, max_value=1.0, value=0.05)

    st.markdown('<span class="input-label">Outlet Location Type</span>', unsafe_allow_html=True)
    Outlet_Location_Type = st.selectbox('', ['Tier 1', 'Tier 2', 'Tier 3'])

    st.markdown('<span class="input-label">Outlet Age</span>', unsafe_allow_html=True)
    Outlet_Age = st.number_input('', min_value=0, max_value=50, value=10)

    # Submit button with full width
    submitted = st.form_submit_button("Predict")

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
    
    # Predict sales using the model and apply the inverse Box-Cox transformation
    original_prediction = predict_sales(input_data)
    
    # Display the original sales prediction with customized style
    st.markdown(f'<p class="prediction-output">Predicted Sales: {original_prediction:.2f}</p>', unsafe_allow_html=True)
    
    # Display feedback message
    st.markdown('<p class="feedback-message">Thank you for using the Big Mart Sales Prediction tool!</p>', unsafe_allow_html=True)

    # Reset button
    if st.button('Reset'):
        st.session_state.clear()  # Clear the session state to reset the app
