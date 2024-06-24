import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open('pipe.pkl', 'rb') as file:
    rf = pickle.load(file)

# Load the dataset
data = pd.read_csv("traineddata.csv")

# Ensure the unique values in the dataset for dropdowns
data['IPS'].unique()

st.title("Laptop Price Predictor")

# Brand selection
company = st.selectbox('Brand', data['Company'].unique())

# Type of laptop
type = st.selectbox('Type', data['TypeName'].unique())

# RAM selection
ram = st.selectbox('Ram(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# OS selection
os = st.selectbox('OS', data['OpSys'].unique())

# Weight input
weight = st.number_input('Weight of the laptop')

# Touchscreen selection
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS selection
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size input
screen_size = st.number_input('Screen Size')

# Screen resolution selection
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# CPU selection
cpu = st.selectbox('CPU', data['CPU_name'].unique())

# HDD selection
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD selection
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU selection
gpu = st.selectbox('GPU(in GB)', data['Gpu brand'].unique())

if st.button('Predict Price'):
    # Convert inputs to suitable format
    ppi = None
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])

    ppi = ((X_resolution**2) + (Y_resolution**2))**0.5 / screen_size

    query = np.array([company, type, ram, weight,
                      touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, -1)

    # Ensure the prediction is made correctly
    try:
        prediction = int(np.exp(rf.predict(query)[0]))
        st.title("Predicted price for this laptop could be between " +
                 str(prediction - 1000) + "₹" + " to " + str(prediction + 1000) + "₹")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
