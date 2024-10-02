import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
def main():
    # Set the title of the app
    st.title("Sales Prediction App")

    # Define input fields for user input
    Item_MPR = st.number_input("Item MRP", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)
    Outlet_type = st.number_input("Outlet Type", min_value=0.0, max_value=5.0, value=0.0, step=1.0)
    Outlet_identifier = st.number_input("Outlet Identifier", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
    Outlet_size = st.number_input("Outlet Size", min_value=0.0, max_value=5.0, value=0.0, step=1.0)
    Item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    Outlet_location_type = st.number_input("Outlet Location Type", min_value=0.0, max_value=5.0, value=0.0, step=1.0)
    Outlet_established_year = st.number_input("Outlet Established Year", min_value=1900, max_value=2024, value=2000, step=1)

    # When the user clicks the 'Predict' button, make the prediction
    if st.button("Predict"):
        # Create a numpy array of the inputs
        features = np.array([[Item_MPR, Outlet_type, Outlet_identifier, Outlet_size, Item_visibility, Outlet_location_type, Outlet_established_year]], dtype=np.float32)

        # Predict using the loaded model
        prediction = model.predict(features)[0]
        
        # Display the prediction
        st.success(f"The predicted sales value is: {prediction:.2f}")

if __name__ == '__main__':
    main()
