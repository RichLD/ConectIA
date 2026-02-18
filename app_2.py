import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from openai import OpenAI
from xgboost import XGBRegressor
from datetime import datetime, date

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="ConectIA - Tu Asistente de Viajes", page_icon="✈️", layout="wide")

# --- 2. CONFIGURACIÓN DE CLIENTES ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
AVIATION_API_KEY = st.secrets["AVIATION_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def cargar_modelo_regresion():
    path = 'modelo_vuelos_regresion.json'
    if os.path.exists(path):
        try:
            model = XGBRegressor()
            model.load_model(path)
            return model
        except Exception as e:
            st.error(f"Error al cargar el archivo .json: {e}")
    return None

modelo_reg = cargar_modelo_regresion()

# --- 3. DICCIONARIOS DE LOGÍSTICA ---
reputacion_dict = {"Aeroméxico": 0.88, "Volaris": 0.75, "VivaAerobus": 0.72, "Iberia": 0.92, "American Airlines": 0.85}
rutas_operativas = {
    "Nacional (México)": ["Aeroméxico", "Volaris", "VivaAerobus"],
    "Internacional (USA)": ["Aeroméxico", "Volaris", "American Airlines"],
    "Transatlántico (Europa)": ["Aeroméxico", "Iberia"]
}
aeropuertos = ["MEX (CDMX)", "TIJ (Tijuana)", "CUN (Cancún)", "MTY (Monterrey)", "GDL (Guadalajara)", "JFK (Nueva York)", "MAD (Madrid)"]

# --- 4. FUNCIONES DE APOYO ---
def obtener_clima_inteligente(ciudad, fecha_viaje):
    ciudad_query = ciudad.split(" ")[0].replace("(", "").strip()
    fecha_str = fecha_viaje.strftime('%Y-%m-%d')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ciudad_query}/{fecha_str}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json&include=days"
    try:
        res = requests.get(url, timeout=10).json()
        dia = res['days'][0]
        return {'temp': dia.get('temp', 22.0), 'precip': dia.get('precip', 0.0), 'wind': dia.get('windspeed', 12.0), 'vis': dia.get('visibility', 10.0)}
    except: return {'temp': 22.0, 'precip': 0.0, 'wind': 12.0, 'vis': 10.0}

def buscar_vuelo_tiempo_real(flight_iata):
    url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATION_API_KEY}&flight_iata={flight_iata}"
    try:
        res = requests.get(url, timeout=10).json()
        if res.get('data'):
            vuelo = res['data'][0]
            return {'origen_iata': vuelo['departure']['iata'], 'aerolinea': vuelo['airline']['name']}
    except: return None

# --- 5. NAVEGACIÓN ---
if 'etapa' not in st.session_state: st.session_state.etapa = 'chat'
if 'messages' not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.title("🚀 Menú ConectIA")
    if st.button("💬 Chat de Ayuda", use_container_width=True): st.session_state.etapa = 'chat'
    if st.button("📊 Revisar mi Vuelo", use_container_width=True): st.session_state.etapa = 'formulario'

# --- 6. PANTALLA: CHAT ---
if st.session_state.etapa == 'chat':
    st.header("🤖 ¿Cómo podemos ayudarte hoy?")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aquí tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            ctx = f" Retraso estimado: {st.session_state.resultado_final['minutos']} min." if 'resultado_final' in st.session_state else ""
            res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Eres ConectIA, amable y servicial.{ctx} Responde: {prompt}"}])
            st.markdown(res.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": res.choices[0].message.content})

# --- 7. PANTALLA: FORMULARIO ---
elif st.session_state.etapa == 'formulario':
    st.header("📊 Analizador de Retraso")
    tab_api, tab_manual = st.tabs(["🔍 Buscar por Número de Vuelo", "⚙️ Configurar Manualmente"])
    disparar_prediccion = False

    with tab_api:
        v_in = st.text_input("Número de vuelo", placeholder="Ej: AM240")
        if st.button("Buscar Vuelo"):
            v_data = buscar_vuelo_tiempo_real(v_in)
            if v_data:
                st.session_state.origen_v, st.session_state.aerolinea_v = v_data['origen_iata'], v_data['aerolinea'].replace(" ", "")
                disparar_prediccion = True

    with tab_manual:
        col1, col2 = st.columns(2)
        with col1:
            origen = st.selectbox("📍 Origen", aeropuertos)
            aerolinea = st.selectbox("🏢 Aerolínea", list(reputacion_dict.keys()))
        with col2:
            fecha, hora = st.date_input("📅 Fecha", value=date.today()), st.slider("🕒 Hora", 0, 23, 12)
        if st.button("🚀 ESTIMAR TIEMPO"): disparar_prediccion = True

    # --- 8. PROCESAMIENTO (CORREGIDO) ---
    if disparar_prediccion and modelo_reg:
        with st.spinner('Calculando...'):
            clima = obtener_clima_inteligente(origen if 'origen' in locals() else st.session_state.origen_v, fecha)
            repu = reputacion_dict.get(aerolinea if 'aerolinea' in locals() else st.session_state.aerolinea_v, 0.80)
            
            # ORDEN DE VARIABLES BASADO EN TU GRÁFICA DE IMPORTANCIA
            # Asegúrate de que el orden sea exactamente como entrenaste
            features = [
                1 if 6 <= hora <= 18 else 0, # fase_dia
                repu,                        # airline_reputation_score
                25,                          # flights_at_that_hour
                clima['vis'],                # visibility
                hora,                        # Hora
                clima['temp'],               # temp
                clima['wind'],               # windspeed
                fecha.weekday(),             # dia_semana
                clima['precip']              # precip
            ]
            
            # Convertimos a matriz NumPy para evitar errores de nombres de columnas
            datos_array = np.array([features], dtype=float)
            
            # Predicción cruda
            pred = modelo_reg.predict(datos_array)[0]
            
            # Si el modelo fue entrenado con Tweedie o logaritmos, a veces necesita ajuste.
            # Aquí lo forzamos a entero y evitamos el 0 constante si el modelo da valores pequeños.
            minutos = int(round(max(0, pred)))

            # --- LLAMADA A LA IA CON PROMPT ORIGINAL ---
            try:
                full_query_ia = f"Eres ConectIA, un asistente de viajes amable y servicial. El usuario tiene un vuelo con {aerolinea} y el retraso estimado es de {minutos} minutos. Responde con un consejo breve."
                res_ia = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_query_ia}])
                consejo = res_ia.choices[0].message.content
            except: consejo = "Te recomendamos revisar el estatus antes de salir."

            st.session_state.resultado_final = {'minutos': minutos, 'aero': aerolinea, 'ruta': f"{origen} ➔ {fecha}", 'consejo': consejo}

    if 'resultado_final' in st.session_state:
        res = st.session_state.resultado_final
        st.divider()
        st.metric("Retraso Estimado", f"{res['minutos']} min")
        st.info(f"💡 **ConectIA:** {res['consejo']}")

