import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler (make sure to provide paths to these files)
# Assuming model.pkl and scaler.pkl are the files where the model and scaler are saved
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

# Load the model and scaler
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title of the app
st.title('Crop Recommendation System')

# Input fields for the user
N = st.number_input('Nitrogen (N)', min_value=0, max_value=100, value=90)
P = st.number_input('Phosphorus (P)', min_value=0, max_value=100, value=42)
K = st.number_input('Potassium (K)', min_value=0, max_value=100, value=43)
temperature = st.number_input('Temperature (°C)', value=20.88)
humidity = st.number_input('Humidity (%)', value=82.00)
ph = st.number_input('pH Level', value=6.5)
rainfall = st.number_input('Rainfall (mm)', value=202.93)

# Button for making predictions
if st.button('Recommend Crop'):
    # Prepare input data as a NumPy array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input data
    scaled_input_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input_data)
    
    # Display the prediction
    st.write(f'The recommended crop is: **{prediction[0]}**')
