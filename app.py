import streamlit as st
import pandas as pd
import joblib
import os

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
    # Asume que estas son las columnas de tu dataset
    age = st.sidebar.slider("Edad", 18, 90, 30)
    income = st.sidebar.number_input("Ingresos Mensuales ($)", 500, 20000, 2000)
    credit_amount = st.sidebar.number_input("Monto de Crédito Solicitado ($)", 100, 50000, 5000)
    duration = st.sidebar.selectbox("Duración del crédito (meses)", [6, 12, 18, 24, 36, 48])
    
    data = {'age': age, 'income': income, 'credit_amount': credit_amount, 'duration': duration}
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
    col1, col2 = st.columns(2)
    
    # Predicción Random Forest
    pred_rf = model_rf.predict(df_input)[0]
    
    # Lógica de visualización
    with col1:
        st.subheader("Resultado Random Forest")
        if pred_rf == 1:
            st.success("✅ Riesgo: Bueno (Aprobado)")
        else:
            st.error("⚠️ Riesgo: Malo (Rechazado)")
            
    with col2:
        # Aquí podrías comparar o promediar resultados
        st.info("El modelo RF ha analizado tus datos con precisión.")
        st.balloons()

# Pie de página
st.markdown("---")
st.caption("Desarrollado con Streamlit por un apasionado del Data Science.")
