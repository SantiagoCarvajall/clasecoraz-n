import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo KNN y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Título de la aplicación
st.title("Asistente Cardiaco")
st.subheader("Autor: Alfredo Diaz")

# Instrucciones de uso
st.write("""
Este asistente cardiaco permite predecir si una persona tiene problemas de corazón
en base a dos parámetros: edad y nivel de colesterol. Utilice los deslizadores para
ingresar estos valores y obtendrá una predicción de si tiene un riesgo de enfermedad
cardíaca. Los valores de edad deben estar entre 18 y 80 años, y el colesterol entre
100 y 600 unidades.
""")

# Crear las pestañas
tabs = st.tabs(["Ingreso de Datos", "Resultado"])

# Pestaña 1: Ingreso de Datos
with tabs[0]:
    edad = st.slider('Edad', min_value=18, max_value=80, value=25)
    colesterol = st.slider('Colesterol', min_value=100, max_value=600, value=200)

    # DataFrame con los datos ingresados
    data = pd.DataFrame([[edad, colesterol]], columns=["edad", "colesterol"])

    # Normalizar los datos
    data_normalizada = escalador.transform(data)

    # Botón para hacer la predicción
    if st.button("Predecir"):
        # Realizar la predicción
        prediccion = modelo_knn.predict(data_normalizada)
        
        # Guardar la predicción para mostrarla en la siguiente pestaña
        st.session_state.prediccion = prediccion[0]

# Pestaña 2: Resultado
with tabs[1]:
    if hasattr(st.session_state, 'prediccion'):
        resultado = st.session_state.prediccion
        
        # Mostrar el resultado de la predicción
        if resultado == 0:
            st.subheader("No tiene problemas de corazón")
            st.image("https://www.diagnosticorojas.com.ar/wp-content/uploads/2023/07/Art-62-Corazon-sano-800x600.jpg")
        else:
            st.subheader("Tiene problemas de corazón")
            st.image("https://as01.epimg.net/deporteyvida/imagenes/2017/10/28/portada/1509177885_209365_1509178036_noticia_normal_recorte1.jpg")
