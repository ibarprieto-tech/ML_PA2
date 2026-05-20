Enlace colab:
https://colab.research.google.com/drive/1jLcLr4Cn-uajo8KcqWJyKXDn9L9JMkqJ?usp=sharing

La evaluación es individual y se enfoca en 3 puntos, el procesamiento de datos (Parte 1), el desarrollo de un modelo de machine learning 
(Parte 2) y su despliegue web en Streamlit usando Github (Parte 3). El flujo de trabajo requerido para desarrollar esta evaluación ha sido abordado en las clases. El estudiante puede utilizar GEMINI como asistente para generar código dentro de Google COLAB y si lo requiere puede complementar con cualquier otra inteligencia artificial como por ejemplo, ChatGPT, Claude, etc.
Parte 1. DATASET (10 puntos)
1. Realice una breve descripción de su dataset y que aprenderá a predecir su modelo de machine learning (2 puntos)
2. Trabajar en Google COLAB. El dataset es de libre elección y debe ser accesible desde su cuaderno de código (COLAB). Recuerde usar pandas para cargarlo en la variable “df”. (2 puntos)
3. Realice el análisis exploratorio de datos que considere necesario (por ejemplo, df.info, df.shape, df.describe, etc.). Genere al menos 3 gráficas y explique su importancia para conocer mejor el dataset (3 puntos)
4. Realice el procesamiento de datos que requiera su dataset. Justifique (3 puntos)
Parte 2. MODELO (5 puntos)
5. Realice la división de datos. Considere usar stratify, validación cruzada u otro de forma justificada. (2 punto)
6. Realice el entrenamiento. Genere al menos dos modelos de machine learning. (1 punto)
7. Evalúe al menos dos métricas y coméntelas. (1 punto)
8. Guarde sus modelos en formato “pkl” usando joblib, como se realizó en clase. (1 punto)
Parte 3. DESPLIEGUE (5 puntos)
9. Adjunte en su github 3 archivos y una carpeta:
a. Archivo llamado “anotaciones.txt” donde adjuntará el PROMPT que utilizó para crear su aplicativo.(1 punto)
b. Archivo llamado “app.py”, que contiene el código para crear su aplicativo. (1 punto)
c. Archivo llamado “requirements.txt”. (1 punto)
d. Carpeta llamada “modelos”, que contiene sus modelos, estos archivos se encuentran en formato “pkl”. (1 punto)
10. [IMPORTANTE] Lo único que se presenta en la plataforma de ISIL es el enlace a su página de Streamlit. Por ejemplo: http://irislunes.streamlit.app/. El contenido de su página web debe incluir un enlace a su cuaderno de código COLAB (para entrarcomo lector, es decir, sin permisos de edición), su nombre y código ISIL. (1 punto)
