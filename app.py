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
st.title('Big Mart Sales Prediction')

# Create a form for user input
with st.form("prediction_form"):
    # Input fields for the selected features
    Item_MRP = st.number_input('Item MRP', min_value=0.0, max_value=300.0, value=100.0)
    
    Outlet_Type = st.selectbox('Outlet Type', ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
    Outlet_Identifier = st.selectbox('Outlet Identifier', ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049'])
    
    Outlet_Size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
    
    Item_Visibility_Interpolate = st.number_input('Item Visibility Interpolate', min_value=0.0, max_value=1.0, value=0.05)
    
    Outlet_Location_Type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    
    Outlet_Age = st.number_input('Outlet Age', min_value=0, max_value=50, value=10)
    
    # Submit button
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
        
        # Display the original sales prediction
        st.write(f"Predicted Sales: {original_prediction:.2f}")
