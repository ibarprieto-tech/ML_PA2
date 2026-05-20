Proyecto: Clasificador de Riesgo Crediticio (PA2)
Descripción
Este aplicativo web, desarrollado con Streamlit, implementa un modelo de Machine Learning diseñado para evaluar la probabilidad de riesgo crediticio de los clientes. El proyecto forma parte de la segunda evaluación parcial (PA2) del curso de Ciencia de Datos en el ISIL.

Tecnologías Utilizadas
Python (Pandas, Scikit-learn, Joblib)

Streamlit (Despliegue web)

Google Colab (Entrenamiento del modelo)

GitHub (Control de versiones)

Enlace a la Aplicación
Acceda al clasificador de riesgo aquí

Notas Técnicas y Limitaciones
Actualmente, el modelo se encuentra en una fase de optimización. Se han identificado los siguientes puntos clave:

Arquitectura del Modelo: Durante el entrenamiento en Colab, el dataset original fue procesado mediante One-Hot Encoding, resultando en un espacio de características de ~50 variables.

Limitación en el despliegue: La versión actual del aplicativo en Streamlit procesa únicamente 4 variables de entrada. El resto de las características se están cargando por defecto, lo que genera un sesgo que resulta en la aprobación automática de todas las solicitudes.

Propuesta de Mejora: La próxima iteración del proyecto consistirá en integrar el pipeline completo de preprocesamiento en el app.py, garantizando que el modelo reciba la totalidad de las variables en la estructura exacta utilizada durante el entrenamiento, eliminando así el sesgo de inferencia.

Fuente de Datos
El dataset utilizado es credit_customers.csv.

https://www.kaggle.com/datasets/ppb00x/credit-risk-customers

El archivo con el Script desarrollado en Colab es:

https://colab.research.google.com/drive/1jLcLr4Cn-uajo8KcqWJyKXDn9L9JMkqJ?usp=sharingEnlace colab:


Créditos
Autor: Ibar Iván Prieto Desulovich

Curso: Ciencia de Datos - ISIL


La evaluación es individual y se enfoca en 3 puntos, el procesamiento de datos (Parte 1), el desarrollo de un modelo de machine learning 
(Parte 2) y su despliegue web en Streamlit usando Github (Parte 3). El flujo de trabajo requerido para desarrollar esta evaluación ha sido abordado en las clases. El estudiante puede utilizar GEMINI como asistente para generar código dentro de Google COLAB y si lo requiere puede complementar con cualquier otra inteligencia artificial como por ejemplo, ChatGPT, Claude, etc.
ISIL. (1 punto)
