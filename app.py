import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- STEP 1: LOAD & TRAIN (The "Brain" of the calculator) ---
@st.cache_resource # This keeps the model in memory so it doesn't re-train every click
def train_model():
    # Replace this with your actual CSV path
    df = pd.read_csv('corrosion_data.csv') 
    X = df[['Sulfur', 'TAN', 'Temp']].values
    y = df['CR'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=5000, random_state=1)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# --- STEP 2: CREATE THE USER INTERFACE (The "Calculator" look) ---
st.title("🧪 Corrosion Rate Calculator")
st.write("Enter the parameters below to predict the Corrosion Rate (CR).")

# Create three columns for the inputs
col1, col2, col3 = st.columns(3)

with col1:
    sulfur = st.number_input("Sulfur Content", min_value=0.1, max_value=1.0, value=0.2, step=0.1)

with col2:
    tan = st.number_input("TAN (Acid Number)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

with col3:
    temp = st.number_input("Temperature (°F/C)", min_value=100, max_value=1000, value=500, step=10)

# --- STEP 3: PREDICTION LOGIC ---
if st.button("Calculate CR"):
    # Format the input for the model
    input_data = np.array([[sulfur, tan, temp]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display result
    st.success(f"### Predicted Corrosion Rate: {prediction[0]:.2f}")
    
    # Simple logic warning
    if prediction[0] > 15:
        st.warning("⚠️ High Corrosion Risk Detected!")