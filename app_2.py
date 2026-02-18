import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from openai import OpenAI
from xgboost import XGBRegressor
from datetime import datetime, date

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="ConectIA - Simulador de Vuelos", page_icon="✈️", layout="wide")

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

# --- 5. NAVEGACIÓN ---
if 'etapa' not in st.session_state: st.session_state.etapa = 'chat'
if 'messages' not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.title("🚀 Menú ConectIA")
    if st.button("💬 Chat de Ayuda", use_container_width=True): st.session_state.etapa = 'chat'
    if st.button("📊 Revisar mi Vuelo", use_container_width=True): st.session_state.etapa = 'formulario'

# --- 6. PANTALLA: CHAT --- (Se mantiene igual)
if st.session_state.etapa == 'chat':
    st.header("🤖 ¿Cómo podemos ayudarte hoy?")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Escribe aquí tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            ctx = f" Retraso estimado: {st.session_state.resultado_final['minutos']} min." if 'resultado_final' in st.session_state else ""
            res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Eres ConectIA, un asistente de viajes amable y servicial.{ctx} Responde: {prompt}"}])
            st.markdown(res.choices[0].message.content)
            st.session_state.messages.append({"role": "assistant", "content": res.choices[0].message.content})

# --- 7. PANTALLA: FORMULARIO ---
elif st.session_state.etapa == 'formulario':
    st.header("📊 Analizador de Retraso")
    
    # Mantenemos los Tabs originales
    tab_manual, tab_clima = st.tabs(["⚙️ Datos del Vuelo", "☁️ Ajustes de Clima (Simulación)"])

    with tab_manual:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                origen = st.selectbox("📍 ¿De dónde sales?", aeropuertos)
                destino = st.selectbox("🏁 ¿A dónde vas?", aeropuertos, index=1)
                
                # Lógica de rutas que ya tenías
                if "MAD" in origen or "MAD" in destino: opciones_aero = rutas_operativas["Transatlántico (Europa)"]
                elif "JFK" in origen or "JFK" in destino: opciones_aero = rutas_operativas["Internacional (USA)"]
                else: opciones_aero = rutas_operativas["Nacional (México)"]
                
                aerolinea = st.selectbox("🏢 Aerolínea", opciones_aero)
            
            with col2:
                fecha = st.date_input("📅 ¿Qué día viajas?", value=date.today())
                hora = st.slider("🕒 ¿A qué hora sale tu vuelo?", 0, 23, 12)

    with tab_clima:
        st.info("Aquí puedes mover las variables para ver cómo afectan al retraso en tiempo real.")
        c_c1, c_c2 = st.columns(2)
        # Obtenemos clima base por API pero permitimos modificarlo
        clima_base = obtener_clima_inteligente(origen, fecha)
        
        sim_temp = c_c1.slider("Temperatura (°C)", -10.0, 45.0, float(clima_base['temp']))
        sim_precip = c_c1.slider("Precipitación (mm)", 0.0, 50.0, float(clima_base['precip']))
        sim_wind = c_c2.slider("Viento (km/h)", 0.0, 100.0, float(clima_base['wind']))
        sim_vis = c_c2.slider("Visibilidad (km)", 0.0, 20.0, float(clima_base['vis']))

    if st.button("🚀 CALCULAR PREDICCIÓN CON ESTOS DATOS", use_container_width=True):
        if modelo_reg:
            with st.spinner('Procesando simulación...'):
                repu = reputacion_dict.get(aerolinea, 0.80)
                
                # ORDEN ESTRICTO: [fase_dia, reputation, flights, visibility, Hora, temp, wind, dia_semana, precip]
                features = [
                    1 if 6 <= hora <= 18 else 0, # fase_dia
                    repu,                        # airline_reputation_score
                    25,                          # flights_at_that_hour
                    sim_vis,                     # visibility
                    hora,                        # Hora
                    sim_temp,                    # temp
                    sim_wind,                    # windspeed
                    fecha.weekday(),             # dia_semana
                    sim_precip                   # precip
                ]
                
                datos_array = np.array([features], dtype=float)
                pred = modelo_reg.predict(datos_array)[0]
                minutos = int(round(max(0, pred)))

                # Prompt de la IA original
                try:
                    prompt_ia = (f"Eres ConectIA, un asistente de viajes amable y servicial. "
                                 f"El usuario tiene un vuelo con {aerolinea} de {origen} a {destino} "
                                 f"y el retraso estimado es de {minutos} minutos. Responde con un consejo breve.")
                    res_ia = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_ia}])
                    consejo = res_ia.choices[0].message.content
                except:
                    consejo = "Llega con tiempo al aeropuerto."

                st.session_state.resultado_final = {'minutos': minutos, 'aero': aerolinea, 'ruta': f"{origen} ➔ {destino}", 'consejo': consejo}

    # --- 9. DESPLIEGUE FINAL ---
    if 'resultado_final' in st.session_state:
        res = st.session_state.resultado_final
        st.divider()
        st.subheader(f"🛡️ Diagnóstico de ConectIA: {res['aero']}")
        col_m1, col_m2 = st.columns([1, 2])
        col_m1.metric("Retraso Estimado", f"{res['minutos']} min")
        
        if res['minutos'] > 30: col_m2.error(f"Se prevé un retraso de {res['minutos']} min.")
        else: col_m2.success("El vuelo se mantiene con buena puntualidad.")
        
        st.info(f"💡 **Consejo:** {res['consejo']}")


