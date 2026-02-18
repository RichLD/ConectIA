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
    page_title="ConectIA - Tu Asistente de Viajes",
    page_icon="✈️",
    layout="wide"
)

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

# --- 4. FUNCIONES DE APOYO ---
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

# --- 5. NAVEGACIÓN ---
if 'etapa' not in st.session_state: st.session_state.etapa = 'chat'
if 'messages' not in st.session_state: st.session_state.messages = []

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
        st.write("Soy ConectIA, tu asistente personal. ¿Tienes dudas sobre tu retraso o necesitas consejos de viaje?")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aquí tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            ctx = ""
            if 'resultado_final' in st.session_state:
                r = st.session_state.resultado_final
                ctx = f" El usuario tiene un retraso de {r['minutos']} min con {r['aero']}."
            
            try:
                full_query = f"Eres ConectIA, un asistente de viajes amable y servicial.{ctx} Responde: {prompt}"
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_query}])
                res_text = response.choices[0].message.content
                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text})
            except: st.error("Error en la conexión con la IA.")

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
                
                if "MAD" in origen or "MAD" in destino: opciones = rutas_operativas["Transatlántico (Europa)"]
                elif "JFK" in origen or "JFK" in destino: opciones = rutas_operativas["Internacional (USA)"]
                else: opciones = rutas_operativas["Nacional (México)"]
                
                aerolinea = st.selectbox("🏢 Aerolínea", opciones)
            
            with col2:
                fecha = st.date_input("📅 Fecha de salida", value=date.today())
                hora = st.slider("🕒 Hora de salida", 0, 23, 12)

    with tab_clima:
        st.info("Mueve estos valores para ver cómo impactan en la predicción del modelo.")
        clima_real = obtener_clima_inteligente(origen, fecha)
        
        c_c1, c_c2 = st.columns(2)
        s_temp = c_c1.slider("Temperatura (°C)", -10.0, 45.0, float(clima_real['temp']))
        s_precip = c_c1.slider("Precipitación (mm)", 0.0, 50.0, float(clima_real['precip']))
        s_wind = c_c2.slider("Viento (km/h)", 0.0, 100.0, float(clima_real['wind']))
        s_vis = c_c2.slider("Visibilidad (km)", 0.0, 20.0, float(clima_real['vis']))

    if st.button("🚀 CALCULAR PREDICCIÓN", use_container_width=True):
        if modelo_reg:
            with st.spinner('Analizando variables...'):
                repu = reputacion_dict.get(aerolinea, 0.80)
                
                # ORDEN ESTRICTO SEGÚN ENTRENAMIENTO
                features = [
                    1 if 6 <= hora <= 18 else 0, # fase_dia
                    repu,                        # airline_reputation_score
                    25,                          # flights_at_that_hour
                    s_vis,                       # visibility
                    hora,                        # Hora
                    s_temp,                      # temp
                    s_wind,                      # windspeed
                    fecha.weekday(),             # dia_semana
                    s_precip                     # precip
                ]
                
                # Predicción usando NumPy para evitar errores de nombres de columnas
                datos_array = np.array([features], dtype=float)
                pred_raw = modelo_reg.predict(datos_array)[0]
                
                # Lógica de ajuste: Si el modelo predice en fracciones de hora, multiplicamos.
                # Si las condiciones son críticas y el resultado es muy bajo (< 5), asumimos escala de horas.
                minutos = pred_raw
                if (s_precip > 10 or s_vis < 5 or s_wind > 40) and minutos < 10:
                    minutos = minutos * 60 
                
                minutos_final = int(max(0, round(minutos)))

                # Prompt original de ConectIA
                try:
                    p_ia = (f"Eres ConectIA, un asistente de viajes amable y servicial. "
                            f"El usuario vuela con {aerolinea} de {origen} a {destino} "
                            f"y el retraso estimado es de {minutos_final} minutos. Dame un consejo breve.")
                    r_ia = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": p_ia}])
                    consejo = r_ia.choices[0].message.content
                except: consejo = "Llega con tiempo al aeropuerto."

                st.session_state.resultado_final = {
                    'minutos': minutos_final, 
                    'aero': aerolinea, 
                    'ruta': f"{origen} ➔ {destino}", 
                    'consejo': consejo
                }

    # --- 8. DESPLIEGUE DE RESULTADOS ---
    if 'resultado_final' in st.session_state:
        res = st.session_state.resultado_final
        st.divider()
        st.subheader(f"🛡️ Diagnóstico ConectIA para {res['aero']}")
        
        col_res1, col_res2 = st.columns([1, 2])
        col_res1.metric("Retraso Estimado", f"{res['minutos']} min")
        
        if res['minutos'] > 45: col_res2.error(f"Se espera un retraso importante en {res['ruta']}.")
        elif res['minutos'] > 15: col_res2.warning(f"Se detecta una demora leve para {res['ruta']}.")
        else: col_res2.success(f"Todo indica que el vuelo en {res['ruta']} será puntual.")
        
        st.info(f"💡 **ConectIA dice:** {res['consejo']}")
