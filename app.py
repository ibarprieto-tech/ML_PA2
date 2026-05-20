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

st.title("💳 Predictor Inteligente de Riesgo Crediticio")

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
    st.write("Procesando predicción...") # Mensaje para confirmar que el botón funciona
    
    # Obtenemos las 48 columnas que espera el modelo
    columnas_modelo = model_rf.feature_names_in_

    # Creamos un DataFrame base lleno de ceros
    df_final = pd.DataFrame(0, index=[0], columns=columnas_modelo)

    # Llenamos con los datos del usuario
    # Nos aseguramos de que existan en el modelo para evitar errores de llave
    for col in ['age', 'duration', 'credit_amount']:
        if col in df_final.columns:
            df_final[col] = df_input[col].iloc[0]
    
    try:
        # Realizamos la predicción
        pred_rf = model_rf.predict(df_final)[0]
        
        st.write("¡Predicción completada!") # Mensaje de éxito
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Resultado Random Forest")
            if pred_rf == 1:
                st.success("✅ Riesgo: Bueno (Aprobado)")
            else:
                st.error("⚠️ Riesgo: Malo (Rechazado)")
        with col2:
            st.info("El modelo ha analizado tus datos.")
            st.balloons()
            
    except Exception as e:
        st.error(f"Error técnico durante la predicción: {e}")
        st.write("Detalles del error para depuración:")
        st.write(e)
