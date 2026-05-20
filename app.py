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
    # 1. Asegúrate de capturar el input en la variable 'credit_amount'
    age = st.sidebar.slider("Edad", 18, 90, 30)
    income = st.sidebar.number_input("Ingresos Mensuales ($)", 500, 20000, 2000)
    
    # Esta es la línea que posiblemente te falta o tiene un nombre distinto:
    credit_amount = st.sidebar.number_input("Monto de Crédito Solicitado ($)", 100, 50000, 5000)
    
    duration = st.sidebar.selectbox("Duración del crédito (meses)", [6, 12, 18, 24, 36, 48])
    
    # 2. Ahora el diccionario usará la variable que definiste arriba
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

        # 1. Creamos un DataFrame con los datos de entrada
    data = {
            'duration': [duration],
            'credit_amount': [credit_amount],
            'age': [age],
            # ... añade aquí el resto de variables que usaste en el entrenamiento
        }
    df_input = pd.DataFrame(data)
        
        # 2. Aplicamos get_dummies para que las columnas coincidan con las 48 que espera el modelo
        # IMPORTANTE: Debes usar las mismas columnas que se generaron en el entrenamiento
    df_final = pd.get_dummies(df_input)
        
        # 3. Alineamos las columnas: obligamos a que el DF tenga exactamente las 48 columnas del modelo
        # Si una columna no existe en el input, se rellena con 0
        columnas_modelo = model_rf.feature_names_in_
        df_final = df_final.reindex(columns=columnas_modelo, fill_value=0)
        
    try:
        # Intentamos predecir
        pred_rf = model_rf.predict(df_final)[0]
        
        # Mostrar resultado
        if pred_rf == 1:
            st.success("✅ Riesgo: Bueno (Aprobado)")
        else:
            st.error("⚠️ Riesgo: Malo (Rechazado)")
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.write("Revisa los logs de Streamlit para ver el detalle técnico.")
