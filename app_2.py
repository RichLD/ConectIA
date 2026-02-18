import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from openai import OpenAI
from xgboost import XGBRegressor
from datetime import datetime, date

# =================================================================
# 1. CONFIGURACIÓN INICIAL Y ESTILO
# =================================================================
st.set_page_config(
    page_title="ConectIA - Hub de Viajes",
    page_icon="✈️",
    layout="wide"
)

# --- CLIENTES API ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    path = 'modelo_vuelos_regresion.json'
    if os.path.exists(path):
        try:
            model = XGBRegressor()
            model.load_model(path)
            return model
        except Exception as e:
            st.error(f"Error técnico al cargar el modelo: {e}")
    return None

modelo_reg = cargar_modelo()

# =================================================================
# 2. LOGÍSTICA Y REGLAS DE NEGOCIO
# =================================================================
reputacion_dict = {
    "Aeroméxico": 0.88, "Volaris": 0.75, "VivaAerobus": 0.72, 
    "Iberia": 0.92, "American Airlines": 0.85
}

rutas_operativas = {
    "Nacional": ["Aeroméxico", "Volaris", "VivaAerobus"],
    "USA": ["Aeroméxico", "Volaris", "American Airlines"],
    "Europa": ["Aeroméxico", "Iberia"]
}

aeropuertos = ["MEX (CDMX)", "TIJ (Tijuana)", "CUN (Cancún)", "MTY (Monterrey)", "GDL (Guadalajara)", "JFK (Nueva York)", "MAD (Madrid)"]

# Inicialización de historial de chat y resultados
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'resultado_final' not in st.session_state:
    st.session_state.resultado_final = None

# =================================================================
# 3. FUNCIONES DE DATOS
# =================================================================
def obtener_clima_real(ciudad, fecha_viaje):
    ciudad_query = ciudad.split(" ")[0].replace("(", "").strip()
    fecha_str = fecha_viaje.strftime('%Y-%m-%d')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ciudad_query}/{fecha_str}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json&include=days"
    try:
        res = requests.get(url, timeout=5).json()
        dia = res['days'][0]
        return {
            'temp': dia.get('temp', 22.0), 
            'precip': dia.get('precip', 0.0), 
            'wind': dia.get('windspeed', 12.0), 
            'vis': dia.get('visibility', 10.0),
            'status': 'ok'
        }
    except:
        return {'temp': 22.0, 'precip': 0.0, 'wind': 12.0, 'vis': 15.0, 'status': 'error'}

# =================================================================
# 4. INTERFAZ DE USUARIO (MONOPÁGINA)
# =================================================================
st.title("✈️ ConectIA: Inteligencia Aeronáutica")
st.subheader("Simulador de retrasos y asistente de viaje en tiempo real")
st.markdown("---")

# División en dos columnas principales
col_izq, col_der = st.columns([1, 1], gap="large")

# -----------------------------------------------------------------
# COLUMNA IZQUIERDA: FORMULARIO Y RESULTADOS
# -----------------------------------------------------------------
with col_izq:
    st.markdown("### 📋 Configuración del Vuelo")
    
    with st.container(border=True):
        # Selección de Origen y Destino
        c1, c2 = st.columns(2)
        with c1:
            origen = st.selectbox("📍 Ciudad de Origen", aeropuertos)
        with c2:
            destino = st.selectbox("🏁 Ciudad de Destino", aeropuertos, index=1)
        
        # VALIDACIÓN 1: No viajar al mismo aeropuerto
        mismo_lugar = (origen == destino)
        if mismo_lugar:
            st.error("❌ El origen y el destino no pueden ser iguales.")
