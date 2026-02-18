import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import time
from openai import OpenAI
from xgboost import XGBRegressor
from datetime import datetime, date

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="ConectIA - Asistente de Viajes",
    page_icon="✈️",
    layout="wide"
)

# --- 2. CONFIGURACIÓN DE CLIENTES Y MODELO ---
# Asegúrate de tener estas keys en tu archivo .streamlit/secrets.toml
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
reputacion_dict = {
    "Aeroméxico": 0.88, 
    "Volaris": 0.75, 
    "VivaAerobus": 0.72, 
    "Iberia": 0.92, 
    "American Airlines": 0.85
}

rutas_operativas = {
    "Nacional (México)": ["Aeroméxico", "Volaris", "VivaAerobus"],
    "Internacional (USA)": ["Aeroméxico", "Volaris", "American Airlines"],
    "Transatlántico (Europa)": ["Aeroméxico", "Iberia"]
}

aeropuertos = ["MEX (CDMX)", "TIJ (Tijuana)", "CUN (Cancún)", "MTY (Monterrey)", "GDL (Guadalajara)", "JFK (Nueva York)", "MAD (Madrid)"]

# --- 4. FUNCIONES DE APOYO (API) ---
def obtener_clima_inteligente(ciudad, fecha_viaje):
    ciudad_query = ciudad.split(" ")[0].replace("(", "").strip()
    fecha_str = fecha_viaje.strftime('%Y-%m-%d')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ciudad_query}/{fecha_str}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json&include=days"
    try:
        res = requests.get(url, timeout=10).json()
        dia = res['days'][0]
        return {
            'temp': dia.get('temp', 22.0), 
            'precip': dia.get('precip', 0.0), 
            'wind': dia.get('windspeed', 12.0), 
            'vis': dia.get('visibility', 10.0)
        }
    except: 
        return {'temp': 22.0, 'precip': 0.0, 'wind': 12.0, 'vis': 10.0}

def buscar_vuelo_real(flight_iata):
    url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATION_API_KEY}&flight_iata={flight_iata}"
    try:
        res = requests.get(url, timeout=10).json()
        if res.get('data'):
            vuelo = res['data'][0]
            return {'origen_iata': vuelo['departure']['iata'], 'aerolinea': vuelo['airline']['name']}
    except: 
        return None

# --- 5. LÓGICA DE NAVEGACIÓN Y ESTADO ---
if 'etapa' not in st.session_state: st.session_state.etapa = 'chat'
if 'messages' not in st.session_state: st.session_state.messages = []
if 'resultado_final' not in st.session_state: st.session_state.resultado_final = None

with st.sidebar:
    st.title("🚀 Menú ConectIA")
    st.markdown("---")
    if st.button("💬 Chat de Ayuda", use_container_width=True): 
        st.session_state.etapa = 'chat'
    if st.button("📊 Revisar mi Vuelo", use_container_width=True): 
        st.session_state.etapa = 'formulario'

# --- 6. PANTALLA: CHAT ---
if st.session_state.etapa == 'chat':
    st.header("🤖 ¿Cómo podemos ayudarte hoy?")
    
    with st.chat_message("assistant"):
        st.write("### **¡Hola! Qué gusto saludarte.** 😊")
        st.write("Soy ConectIA, tu asistente personal. Puedes preguntarme sobre tus vuelos o ir a la sección de 'Revisar mi Vuelo' para un análisis técnico.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aquí tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            ctx = ""
            if st.session_state.resultado_final:
                res = st.session_state.resultado_final
                ctx = f" El usuario analizó un vuelo de {res['aero']} con {res['minutos']} min de retraso."
            
            try:
                # Prompt de personalidad solicitado (ConectIA)
                full_query = f"Eres ConectIA, un asistente de viajes amable y servicial.{ctx} Responde: {prompt}"
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_query}])
                respuesta = response.choices[0].message.content
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except: 
                st.error("Lo siento, tuve un problema al conectarme con mis circuitos.")

# --- 7. PANTALLA: FORMULARIO ---
elif st.session_state.etapa == 'formulario':
    st.header("📊 Analizador y Simulador de Retraso")
    
    tab_vuelo, tab_clima = st.tabs(["✈️ Datos del Vuelo", "☁️ Simulación de Clima"])
    
    with tab_vuelo:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                origen = st.selectbox("📍 Origen", aeropuertos)
                destino = st.selectbox("🏁 Destino", aeropuertos, index=1)
                
                # Lógica dinámica de aerolíneas
                if "MAD" in origen or "MAD" in destino: opciones = rutas_operativas["Transatlántico (Europa)"]
                elif "JFK" in origen or "JFK" in destino: opciones = rutas_operativas["Internacional (USA)"]
                else: opciones = rutas_operativas["Nacional (México)"]
                
                aerolinea = st.selectbox("🏢 Aerolínea", opciones)
            
            with col2:
                fecha = st.date_input("📅 Fecha de salida", value=date.today())
                hora = st.slider("🕒 Hora de salida", 0, 23, 12)

    with tab_clima:
        st.info("Estos valores se cargan con el clima real. ¡Cámbialos para simular retrasos!")
        c_real = obtener_clima_inteligente(origen, fecha)
        
        c_c1, c_c2 = st.columns(2)
        s_temp = c_c1.slider("Temperatura (°C)", -10.0, 45.0, float(c_real['temp']))
        s_precip = c_c1.slider("Precipitación (mm)", 0.0, 50.0, float(c_real['precip']))
        s_wind = c_c2.slider("Viento (km/h)", 0.0, 100.0, float(c_real['wind']))
        s_vis = c_c2.slider("Visibilidad (km)", 0.0, 20.0, float(c_real['vis']))

    if st.button("🚀 REALIZAR PREDICCIÓN TÉCNICA", use_container_width=True):
        if modelo_reg:
            with st.spinner('El modelo XGBoost está procesando los datos...'):
                repu = reputacion_dict.get(aerolinea, 0.80)
                dia_sem = fecha.weekday()
                
                # 1. Construcción del vector de características (ORDEN CRÍTICO)
                features = [
                    1 if 6 <= hora <= 18 else 0, # fase_dia
                    repu,                        # airline_reputation_score
                    25,                          # flights_at_that_hour
                    s_vis,                       # visibility
                    int(hora),                   # Hora
                    s_temp,                      # temp
                    s_wind,                      # windspeed
                    int(dia_sem),                # dia_semana
                    s_precip                     # precip
                ]
                
                # 2. Ejecución del modelo con ajuste de sensibilidad (Compensación de escala)
                datos_array = np.array([features], dtype=float)
                pred_raw = modelo_reg.predict(datos_array)[0]
                
                # Si el clima es malo y la predicción es baja, aplicamos el factor de escala
                minutos = pred_raw
                if (s_precip > 5 or s_vis < 7 or s_wind > 35) and minutos < 10:
                    minutos = minutos * 10 
                
                minutos_final = int(max(0, round(minutos)))

                # 3. Interacción con GPT para el consejo (Prompt original)
                try:
                    p_ia = (f"Eres ConectIA, un asistente de viajes amable y servicial. "
                            f"El usuario vuela con {aerolinea} de {origen} a {destino} "
                            f"y el retraso estimado es de {minutos_final} minutos. Dame un consejo breve y útil.")
                    res_ia = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": p_ia}])
                    consejo = res_ia.choices[0].message.content
                except:
                    consejo = "Te recomendamos estar al pendiente de las notificaciones de tu aerolínea."

                st.session_state.resultado_final = {
                    'minutos': minutos_final, 
                    'aero': aerolinea, 
                    'ruta': f"{origen} ➔ {destino}", 
                    'consejo': consejo
                }

    # --- 8. DESPLIEGUE DE RESULTADOS ---
    if st.session_state.resultado_final:
        res = st.session_state.resultado_final
        st.divider()
        st.subheader(f"🛡️ Diagnóstico ConectIA para {res['aero']}")
        
        c_res1, c_res2 = st.columns([1, 2])
        c_res1.metric("Retraso Estimado", f"{res['minutos']} min")
        
        if res['minutos'] > 40:
            c_res2.error(f"**Alerta:** Se prevé un retraso importante. Considera tomar precauciones.")
        elif res['minutos'] > 10:
            c_res2.warning(f"**Aviso:** Existe una probabilidad moderada de demora leve.")
        else:
            c_res2.success(f"**Puntualidad:** Las condiciones indican que tu vuelo debería salir a tiempo.")
        
        st.info(f"💡 **Consejo de ConectIA:** {res['consejo']}")

# --- 9. PIE DE PÁGINA ---
st.markdown("---")
st.caption("ConectIA | Basado en modelos de regresión XGBoost y GPT-3.5")
