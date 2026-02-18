import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from openai import OpenAI
from xgboost import XGBRegressor
from datetime import datetime, date

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="ConectIA - Estimador de Retraso", page_icon="⏱️", layout="wide")

# --- 2. CONFIGURACIÓN DE CLIENTES ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def cargar_modelo_regresion():
    path = 'modelo_vuelos_regresion.json'
    if os.path.exists(path):
        model = XGBRegressor()
        model.load_model(path)
        return model
    return None

modelo_reg = cargar_modelo_regresion()

# --- 3. DICCIONARIOS DE LOGÍSTICA ---
reputacion_dict = {"Aeroméxico": 0.88, "Volaris": 0.75, "VivaAerobus": 0.72, "Iberia": 0.92, "American Airlines": 0.85}
aeropuertos = ["MEX (CDMX)", "TIJ (Tijuana)", "CUN (Cancún)", "MTY (Monterrey)", "GDL (Guadalajara)", "JFK (Nueva York)", "MAD (Madrid)"]

# --- 4. FUNCIONES DE APOYO ---
def obtener_clima(ciudad, fecha_viaje):
    ciudad_q = ciudad.split(" ")[0].replace("(", "").strip()
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ciudad_q}/{fecha_viaje}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json&include=days"
    try:
        res = requests.get(url, timeout=10).json()
        dia = res['days'][0]
        return {'temp': dia['temp'], 'precip': dia['precip'], 'wind': dia['windspeed'], 'vis': dia['visibility']}
    except:
        return {'temp': 22.0, 'precip': 0.0, 'wind': 12.0, 'vis': 15.0}

# --- 5. INTERFAZ ---
st.title("⏱️ Estimador de Tiempo de Retraso con ConectIA")

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        origen = st.selectbox("📍 Origen", aeropuertos)
        aerolinea = st.selectbox("🏢 Aerolínea", list(reputacion_dict.keys()))
    with col2:
        fecha = st.date_input("📅 Fecha", value=date.today())
        hora = st.slider("🕒 Hora de salida (0-23h)", 0, 23, 12)

if st.button("🚀 ESTIMAR MINUTOS DE RETRASO", use_container_width=True):
    if modelo_reg is None:
        st.error("No se pudo cargar 'modelo_vuelos_regresion.json'. Revisa tu repositorio de GitHub.")
    else:
        with st.spinner('Analizando condiciones...'):
            clima = obtener_clima(origen, fecha)
            
            # ORDEN EXACTO basado en tu entrenamiento
            input_dict = {
                'fase_dia': 1 if 6 <= hora <= 18 else 0,
                'airline_reputation_score': reputacion_dict[aerolinea],
                'flights_at_that_hour': 25,
                'visibility': clima['vis'],
                'Hora': int(hora),
                'temp': clima['temp'],
                'windspeed': clima['wind'],
                'dia_semana': int(fecha.weekday()),
                'precip': clima['precip']
            }
            
            df_input = pd.DataFrame([input_dict])
            minutos = int(max(0, modelo_reg.predict(df_input)[0]))

            # --- DESPLIEGUE DE RESULTADOS ---
            st.divider()
            c1, c2 = st.columns([1, 2])
            c1.metric("Retraso Estimado", f"{minutos} min")
            
            if minutos > 45:
                c2.error(f"**Alerta:** Se estima una demora importante de {minutos} min en la ruta {origen}.")
            elif minutos > 15:
                c2.warning(f"**Aviso:** Podrías tener un retraso leve de aproximadamente {minutos} min.")
            else:
                c2.success(f"**Puntualidad:** Todo indica que saldrás a tiempo.")

            # --- LLAMADA A LA IA CON EL PROMPT ORIGINAL ---
            try:
                # Mantenemos la estructura de tu prompt original
                full_query = (f"Eres ConectIA, un asistente de viajes amable y servicial. "
                              f"El usuario tiene un vuelo con {aerolinea} desde {origen} "
                              f"y el retraso estimado es de {minutos} minutos. "
                              f"Responde con un consejo breve y amable.")
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": full_query}]
                )
                st.info(f"💡 **ConectIA dice:** {response.choices[0].message
