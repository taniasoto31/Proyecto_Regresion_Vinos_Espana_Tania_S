from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios de Vinos de España",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Predictor de Precios de Vinos de España")
st.write("Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado para predecir el precio de los vinos españoles en función de sus características.")

# 1. Definir la ruta raíz absoluta de tu proyecto en Windows
#project_root = Path(r"C:\F5Proyectos\Proyecto_Regresion_Vinos_Espana_Tania")
#ruta_modelo = project_root / "models" / "modelo_final_vinos.pkl"
# ✅ AHORA (Ruta relativa automática: toma la carpeta actual del proyecto sin importar dónde esté alojado)
project_root = Path(__file__).resolve().parent
ruta_modelo = project_root / "models" / "modelo_final_vinos.pkl"

# 2. Función para cargar el modelo de manera segura y cacheada
@st.cache_resource
def cargar_modelo():
    if ruta_modelo.exists():
        return joblib.load(ruta_modelo)
    else:
        return None

modelo = cargar_modelo()

if modelo is None:
    st.error(f"❌ No se encontró el archivo del modelo en la ruta: {ruta_modelo}")
    st.stop()
else:
    st.success("¡Modelo cargado correctamente!")

st.markdown("---")

# 3. Formulario de entradas para el usuario (Ajusta los nombres según las columnas con las que entrenaste tu modelo)
st.subheader("📝 Introduce las características del vino:")

col1, col2 = st.columns(2)

with col1:
    alcohol = st.slider("Graduación de Alcohol (%)", min_value=8.0, max_value=16.0, value=13.0, step=0.1)
    acidez_volatil = st.number_input("Acidez Volátil", min_value=0.0, max_value=2.0, value=0.35, step=0.01)

with col2:
    sulfatos = st.number_input("Sulfatos", min_value=0.2, max_value=2.0, value=0.6, step=0.05)
    pH = st.slider("Nivel de pH", min_value=2.5, max_value=4.5, value=3.4, step=0.01)

# 4. Botón para realizar la predicción
if st.button("🚀 Calcular Precio Estimado", type="primary"):
    # Creamos un DataFrame con los datos introducidos (asegúrate de que los nombres de las columnas coincidan con X_train)
    # Nota: Si tu modelo usa más columnas, añádelas aquí en el diccionario
    datos_entrada = pd.DataFrame({
        'alcohol': [alcohol],
        'volatile_acidity': [acidez_volatil],
        'sulphates': [sulfatos],
        'pH': [pH]
    })
    
    try:
        # Realizar la predicción
        prediccion = modelo.predict(datos_entrada)[0]
        
        # Mostrar el resultado de forma atractiva
        st.markdown("### 📊 Resultado de la Predicción:")
        st.metric(label="Precio Estimado del Vino", value=f"{prediccion:.2f} €")
        
    except Exception as e:
        st.error(f"Ocurrió un error al hacer la predicción: {e}")
        st.info("💡 Consejo: Asegúrate de que las columnas del DataFrame de entrada coincidan exactamente con las que usó tu modelo durante el entrenamiento.")