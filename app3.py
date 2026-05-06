import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Multi-Material Corrosion Calc", page_icon="🧪")

# --- MULTI-MODEL LOADING LOGIC ---
@st.cache_resource
def train_material_model(material_choice):
    # Mapping selection to filenames
    files = {
        "Carbon Steel": "CSCR.csv",  # or P9_data.csv
        "P9 Material": "P9CR.csv"
    }
    
    file_path = files[material_choice]
    
    try:
        df = pd.read_csv(file_path)
        # Ensure column names match your CSV exactly
        X = df[['Sulfur', 'TAN', 'Temperature']].values
        y = df['CR'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=5000, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading {material_choice}: {e}")
        return None, None

# --- UI DESIGN ---
st.title("🧪 Corrosion Rate Calculator")

# 1. Material Selection
material = st.selectbox("Select Material Type:", ["P9 Material", "P7 Material"])

# Load the specific model for the selection
model, scaler = train_material_model(material)

if model:
    st.info(f"Currently using the model trained for **{material}**")
    
    # 2. Input Fields
    col1, col2, col3 = st.columns(3)
    with col1:
        s = st.number_input("Sulfur", value=0.5, step=0.1)
    with col2:
        t = st.number_input("TAN", value=1.0, step=0.1)
    with col3:
        temp = st.number_input("Temperature", value=500, step=10)

    # 3. Prediction
    if st.button("Predict Corrosion Rate"):
        features = np.array([[s, t, temp]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        st.success(f"### Predicted CR for {material}: {prediction:.2f}")
