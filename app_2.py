import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from openai import OpenAI
from xgboost import XGBRegressor
from datetime import datetime, date

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="ConectIA - Hub de Viajes",
    page_icon="✈️",
    layout="wide"
)

# --- 2. CONFIGURACIÓN DE CLIENTES ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def cargar_modelo():
    path = 'modelo_vuelos_regresion.json'
    if os.path.exists(path):
        try:
            model = XGBRegressor()
            model.load_model(path)
            return model
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
    return None

modelo_reg = cargar_modelo()

# --- 3. DICCIONARIOS ---
reputacion_dict = {"Aeroméxico": 0.88, "Volaris": 0.75, "VivaAerobus": 0.72, "Iberia": 0.92, "American Airlines": 0.85}
aeropuertos = ["MEX (CDMX)", "TIJ (Tijuana)", "CUN (Cancún)", "MTY (Monterrey)", "GDL (Guadalajara)", "JFK (Nueva York)", "MAD (Madrid)"]

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'resultado_final' not in st.session_state:
    st.session_state.resultado_final = None

# --- 4. FUNCIÓN CLIMA ---
def obtener_clima_real(ciudad, fecha_viaje):
    ciudad_query = ciudad.split(" ")[0].replace("(", "").strip()
    fecha_str = fecha_viaje.strftime('%Y-%m-%d')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ciudad_query}/{fecha_str}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json&include=days"
    try:
        res = requests.get(url, timeout=10).json()
        dia = res['days'][0]
        return {'temp': dia.get('temp', 22.0), 'precip': dia.get('precip', 0.0), 'wind': dia.get('windspeed', 12.0), 'vis': dia.get('visibility', 10.0)}
    except: return {'temp': 22.0, 'precip': 0.0, 'wind': 12.0, 'vis': 10.0}

# --- 5. ESTRUCTURA DE PÁGINA ÚNICA ---
st.title("✈️ ConectIA: Centro de Mando de Vuelos")
st.markdown("Analiza tu vuelo y chatea con nuestra IA en una sola pantalla.")

col_form, col_chat = st.columns([1, 1], gap="large")

# --- COLUMNA IZQUIERDA: FORMULARIO ---
with col_form:
    st.subheader("📊 Datos del Viaje")
    with st.container(border=True):
        origen = st.selectbox("📍 Origen", aeropuertos)
        destino = st.selectbox("🏁 Destino", aeropuertos, index=1)
        
        # VALIDACIÓN: Evitar mismo aeropuerto
        if origen == destino:
            st.error("⚠️ El aeropuerto de destino no puede ser el mismo que el de origen.")
            btn_analizar = st.button("🚀 REALIZAR ANÁLISIS", use_container_width=True, disabled=True)
        else:
            btn_analizar = st.button("🚀 REALIZAR ANÁLISIS", use_container_width=True)
            
        aerolinea = st.selectbox("🏢 Aerolínea", list(reputacion_dict.keys()))
        fecha = st.date_input("📅 Fecha", value=date.today())
        hora = st.slider("🕒 Hora de salida", 0, 23, 12)
    if btn_analizar:
        if modelo_reg:
            with st.spinner('Analizando variables climáticas y operativas...'):
                clima = obtener_clima_real(origen, fecha)
                repu = reputacion_dict.get(aerolinea, 0.80)
                
                # Features para el modelo
                features = [1 if 6 <= hora <= 18 else 0, repu, 25, clima['vis'], int(hora), clima['temp'], clima['wind'], int(fecha.weekday()), clima['precip']]
                
                # Predicción y ajuste
                datos_array = np.array([features], dtype=float)
                pred_raw = modelo_reg.predict(datos_array)[0]
                minutos = pred_raw * 10 if (clima['precip'] > 3 or clima['vis'] < 8) and pred_raw < 10 else pred_raw
                minutos_final = int(max(0, round(minutos)))

                st.session_state.resultado_final = {
                    'minutos': minutos_final, 'aero': aerolinea, 'clima': clima,
                    'ruta': f"{origen} ➔ {destino}"
                }

    # Despliegue de Resultados debajo del formulario
    if st.session_state.resultado_final:
        res = st.session_state.resultado_final
        st.markdown("---")
        st.write("### 🛡️ Resultado del Análisis")
        c1, c2 = st.columns(2)
        c1.metric("Retraso Estimado", f"{res['minutos']} min")
        c2.metric("Temp. Detectada", f"{res['clima']['temp']} °C")
        
        if res['minutos'] > 30:
            st.error(f"Se prevé una demora significativa con {res['aero']}.")
        else:
            st.success(f"Condiciones óptimas para la ruta {res['ruta']}.")

# --- COLUMNA DERECHA: CHAT IA ---
with col_chat:
    st.subheader("🤖 Chat con ConectIA")
    
    # Contenedor para los mensajes (con altura fija para scroll)
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        if not st.session_state.messages:
            st.write("¡Hola! Soy ConectIA. Ingresa los datos de tu vuelo a la izquierda para empezar o pregúntame lo que quieras.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu duda aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                # Incluir el contexto del análisis si existe
                ctx = ""
                if st.session_state.resultado_final:
                    r = st.session_state.resultado_final
                    ctx = f" El usuario analizó un vuelo de {r['aero']} con {r['minutos']} min de retraso."
                
                try:
                    full_query = f"Eres ConectIA, amable y servicial.{ctx} Responde: {prompt}"
                    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_query}])
                    respuesta = response.choices[0].message.content
                    st.markdown(respuesta)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                except:
                    st.error("Error de conexión.")
        st.rerun()

st.markdown("---")
st.caption("ConectIA v2.0 | Sistema Unificado de Predicción y Asistencia")

