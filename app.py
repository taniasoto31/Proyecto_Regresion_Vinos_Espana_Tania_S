from pathlib import Path
import joblib
import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios de Vinos de España",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Predictor de Precios de Vinos de España")
st.write("Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado para predecir el precio de los vinos españoles en función de sus características.")

# Función robusta para cargar el modelo y retornar tanto el objeto como la ruta (para depurar si falla)
@st.cache_resource
def cargar_y_obtener_ruta():
    base_dir = Path(__file__).resolve().parent
    
    # Rutas posibles a evaluar en orden de prioridad
    candidatos = [
        base_dir / "models" / "modelo_final_vinos.pkl",
        base_dir / "Models" / "modelo_final_vinos.pkl",
        base_dir / "modelo_final_vinos.pkl"
    ]
    
    for ruta in candidatos:
        if ruta.exists():
            return joblib.load(ruta), ruta
            
    # Si ninguna existe, devolvemos None y la ruta estándar para mostrar el error
    return None, candidatos[0]

# Cargar el modelo
modelo, ruta_intentada = cargar_y_obtener_ruta()

if modelo is None:
    st.error(f"❌ No se encontró el archivo del modelo. Se buscó en: {ruta_intentada}")
    st.info("💡 Asegúrate de que la carpeta 'models' y el archivo 'modelo_final_vinos.pkl' estén subidos a GitHub.")
    st.stop()
else:
    st.success("¡Modelo cargado correctamente!")

st.markdown("---")

# Formulario de entradas para el usuario
st.subheader("📝 Introduce las características del vino:")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.slider("Graduación de Alcohol (%)", min_value=8.0, max_value=16.0, value=13.0, step=0.1)
    acidez_volatil = st.number_input("Acidez Volátil", min_value=0.0, max_value=2.0, value=0.35, step=0.01)

with col2:
    sulfatos = st.number_input("Sulfatos", min_value=0.2, max_value=2.0, value=0.6, step=0.05)
    pH = st.slider("Nivel de pH", min_value=2.5, max_value=4.5, value=3.4, step=0.01)

# Botón para realizar la predicción
if st.button("🚀 Calcular Precio Estimado", type="primary"):
    datos_entrada = pd.DataFrame({
        'alcohol': [alcohol],
        'volatile_acidity': [acidez_volatil],
        'sulphates': [sulfatos],
        'pH': [pH]
    })
    
    try:
        prediccion = modelo.predict(datos_entrada)[0]
        st.markdown("### 📊 Resultado de la Predicción:")
        st.metric(label="Precio Estimado del Vino", value=f"{prediccion:.2f} €")
        
    except Exception as e:
        st.error(f"Ocurrió un error al hacer la predicción: {e}")
        st.info("💡 Consejo: Asegúrate de que las columnas del DataFrame de entrada coincidan exactamente con las que usó tu modelo durante el entrenamiento.")