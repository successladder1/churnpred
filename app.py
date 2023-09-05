import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved preprocessing pipeline
with open('preprocessing_pipeline.pkl', 'rb') as file:
    preprocessing_pipeline = pickle.load(file)

# Load the saved feature selection/classification pipeline
with open('classification_pipeline.pkl', 'rb') as file:
    classification_pipeline = pickle.load(file)

# Streamlit UI
st.title('Churn prediction')
st.write('Enter input data for prediction:')

# Create input fields for user input data
user_input_data = {}
user_input_data['Age'] = st.number_input('Age', min_value=0, max_value=100)
user_input_data['Gender'] = st.selectbox('Gender', ['Male', 'Female'])
user_input_data['Location'] = st.text_input('Location')
user_input_data['Subscription_Length_Months'] = st.number_input('Subscription Length (Months)', min_value=0, max_value=100)
user_input_data['Monthly_Bill'] = st.number_input('Monthly Bill', min_value=0.0, max_value=1000.0)
user_input_data['Total_Usage_GB'] = st.number_input('Total Usage (GB)', min_value=0, max_value=1000)

# Convert user input data to a DataFrame
user_input_df = pd.DataFrame([user_input_data])

# Apply preprocessing using the preprocessing pipeline
preprocessed_data = preprocessing_pipeline.transform(user_input_df)
# Apply feature selection and make predictions using the classification pipeline
prediction = classification_pipeline.predict(preprocessed_data)

# Display the prediction
st.write('Prediction:', prediction[0])
