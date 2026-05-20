import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import sklearn

# Configuración de página
st.set_page_config(page_title="Predicción de Riesgo Crediticio", page_icon="💳", layout="wide")

# Estilos personalizados
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { color: white; background-color: #4CAF50; border-radius: 10px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# Título y Descripción
st.title("💳 Predictor Inteligente de Riesgo Crediticio")
st.markdown("Este aplicativo evalúa la salud crediticia de un cliente utilizando modelos de IA.")

# Barra lateral para entradas
st.sidebar.header("Datos del Cliente")

def user_input_features():
    age = st.sidebar.slider("Edad", 18, 90, 30)
    income = st.sidebar.number_input("Ingresos Mensuales ($)", 500, 20000, 2000)
    credit_amount = st.sidebar.number_input("Monto de Crédito Solicitado ($)", 100, 50000, 5000)
    duration = st.sidebar.selectbox("Duración del crédito (meses)", [6, 12, 18, 24, 36, 48])
    
    data = {
        'age': age, 
        'income': income, 
        'credit_amount': credit_amount, 
        'duration': duration
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Mostrar datos ingresados
st.subheader("Datos Ingresados")
st.table(df_input)

# Cargar Modelos
@st.cache_resource
def load_models():
    path_rf = os.path.join('modelos', 'modelo_rf.pkl')
    path_lr = os.path.join('modelos', 'modelo_lr.pkl')
    rf = joblib.load(path_rf)
    lr = joblib.load(path_lr)
    return rf, lr

model_rf, model_lr = load_models()

# Predicción
if st.button("🚀 Predecir Riesgo"):
    # Obtenemos las 48 columnas que espera el modelo
    columnas_modelo = model_rf.feature_names_in_

    # Creamos un DataFrame base lleno de ceros
    df_final = pd.DataFrame(0, index=[0], columns=columnas_modelo)
