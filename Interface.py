import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

with open('model.pkl', 'rb') as model_file:
    model=pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler=pickle.load(scaler_file)

st.title('Crop Recommendation System')

N=st.number_input('Nitrogen',value=90)
P=st.number_input('Phosphorus',value=42)
K=st.number_input('Potassium',value=43)
Temp=st.number_input('Temperature °C',value=20.88)
Hum=st.number_input('Humidity',value=82.00)
PH=st.number_input('pH Level',value=6.5)
Rain=st.number_input('Rainfall',value=202.93)

if st.button('Recommend'):
    inp=np.array([[N,P,K,Temp,Hum,PH,Rain]])
    inp=scaler.transform(inp)
    pred=model.predict(inp)
    st.write('The recommended crop is:',pred[0])