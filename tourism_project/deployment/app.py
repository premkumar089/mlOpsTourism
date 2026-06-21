
import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# Define the Hugging Face model repository ID
HF_MODEL_REPO_ID = "sdncountry/tourism-package-model"
MODEL_FILENAME = "best_model.joblib"

# Function to download the model
@st.cache_resource
def download_model():
    try:
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=MODEL_FILENAME)
        st.success(f"Model downloaded from Hugging Face Hub to {model_path}")
        return model_path
    except Exception as e:
        st.error(f"Error downloading model from Hugging Face: {e}")
        return None

# Download and load the model
model_file_path = download_model()
if model_file_path:
    try:
        model = joblib.load(model_file_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        model = None
else:
    model = None

st.title("Wellness Tourism Package Purchase Predictor")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

if model is not None:
    # Input fields for all features
    st.header("Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        typeofcontact = st.selectbox("Type of Contact", ['Company Invited', 'Self Inquiry'])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
        preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])

    with col2:
        maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        numberofchildrenvisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Executive', 'AVP', 'VP', 'Director', 'CEO'])

    with col3:
        monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=50000.0, step=1000.0)
        pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
        numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=10.0)

    # Create a DataFrame from inputs
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': typeofcontact,
        'CityTier': citytier,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': numberofpersonvisiting,
        'PreferredPropertyStar': preferredpropertystar,
        'MaritalStatus': maritalstatus,
        'NumberOfTrips': numberoftrips,
        'Passport': passport,
        'OwnCar': owncar,
        'NumberOfChildrenVisiting': numberofchildrenvisiting,
        'Designation': designation,
        'MonthlyIncome': monthlyincome,
        'PitchSatisfactionScore': pitchsatisfactionscore,
        'ProductPitched': productpitched,
        'NumberOfFollowups': numberoffollowups,
        'DurationOfPitch': durationofpitch
    }])

    # Placeholder for 'CustomerID' and 'Unnamed: 0' if the model expects them
    # The train.py script dropped these before training. If the model is a pipeline that expects these,
    # they should be handled. Assuming the trained pipeline will automatically ignore or handle new/missing columns.
    # However, to match the training data columns, we need to ensure the order/presence is consistent.
    # For simplicity, we assume the pipeline handles column selection internally or we pass a DataFrame with expected columns.
    # Let's add dummy CustomerID and Unnamed:0 if the model was trained with them and expects them.
    # If the model pipeline correctly drops them, then no issue.
    # For robust deployment, ensure the input DataFrame columns match those seen during training.
    # A common practice is to have a list of features the model expects and create the DataFrame based on that.

    if st.button("Predict Purchase"): # Button for prediction
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)[:, 1]

            if prediction[0] == 1:
                st.success(f"Prediction: Customer is LIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})")
            else:
                st.info(f"Prediction: Customer is UNLIKELY to purchase the package (Probability: {prediction_proba[0]:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.warning("Model could not be loaded. Please check the model repository and token.")
