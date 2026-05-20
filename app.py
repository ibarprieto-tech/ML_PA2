import joblib
import os
import streamlit as st
import pandas as pd
import numpy as np
import sklearn

# Configuración de página
st.set_page_config(page_title="Predicción de Riesgo Crediticio", page_icon="💳", layout="wide")

# Estilos personalizados para un toque divertido
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { color: white; background-color: #4CAF50; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Título y Descripción
st.title("💳 Predictor Inteligente de Riesgo Crediticio")
st.markdown("""
¡Bienvenido! Este aplicativo evalúa la salud crediticia de un cliente utilizando modelos de IA.
Puedes revisar el desarrollo técnico detallado en mi [Cuaderno de Google Colab](https://colab.research.google.com/tu-enlace-aqui).
""")

# Barra lateral para entradas
st.sidebar.header("Datos del Cliente")

def user_input_features():
    age = st.sidebar.slider("Edad", 18, 90, 30)
    income = st.sidebar.number_input("Ingresos Mensuales ($)", 500, 20000, 2000)
    # ... y así con los demás
    
    # IMPORTANTE: Los nombres aquí deben ser IDÉNTICOS a lo que viste en 
    # "Lo que espera el modelo"
    data = {
        'Age': age, 
        'Income': income, 
        'Credit_amount': credit_amount, 
        'Duration': duration
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

# Mostrar datos ingresados
st.subheader("Datos Ingresados")
st.table(df_input)

# Cargar Modelos desde la subcarpeta 'modelos'
@st.cache_resource
def load_models():
    # Usamos os.path.join para que funcione en cualquier sistema operativo
    path_rf = os.path.join('modelos', 'modelo_rf.pkl')
    path_lr = os.path.join('modelos', 'modelo_lr.pkl')
    
    rf = joblib.load(path_rf)
    lr = joblib.load(path_lr)
    return rf, lr

model_rf, model_lr = load_models()

# Predicción
if st.button("🚀 Predecir Riesgo"):
    # Imprimimos lo que el modelo espera y lo que le enviamos
    st.write("---")
    st.write("Depuración de columnas:")
    st.write("Lo que espera el modelo:", model_rf.feature_names_in_.tolist())
    st.write("Lo que le envío yo:", df_input.columns.tolist())
    
    try:
        # Intentamos predecir
        pred_rf = model_rf.predict(df_input)[0]
        
        # Mostrar resultado
        if pred_rf == 1:
            st.success("✅ Riesgo: Bueno (Aprobado)")
        else:
            st.error("⚠️ Riesgo: Malo (Rechazado)")
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.write("Revisa los logs de Streamlit para ver el detalle técnico.")
