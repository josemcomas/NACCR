import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Corrosion Calc", page_icon="🧪")

@st.cache_resource
def load_and_train():
    # It will look for the csv in the same GitHub folder
    df = pd.read_csv('corrosion_data.csv')
    X = df[['Sulfur', 'TAN', 'Temperature']].values
    y = df['CR'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=5000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_and_train()

st.title("🧪 Corrosion Rate Calculator")
st.write("Enter values to predict the Corrosion Rate (CR) using a Neural Network.")

# Input fields
col1, col2, col3 = st.columns(3)
with col1:
    s = st.number_input("Sulfur", value=0.8)
with col2:
    t = st.number_input("TAN", value=1.0)
with col3:
    temp = st.number_input("Temperature", value=500)

if st.button("Predict"):
    features = np.array([[s, t, temp]])
    prediction = model.predict(scaler.transform(features))[0]
    st.success(f"### Predicted CR: {prediction:.2f}")
