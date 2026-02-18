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

with col_form:
    st.subheader("📊 Configuración del Vuelo")
    with st.container(border=True):
        origen = st.selectbox("📍 ¿De dónde sales?", aeropuertos)
        destino = st.selectbox("🏁 ¿A dónde vas?", aeropuertos, index=1)
        
        # Validación 1: No viajar al mismo aeropuerto
        mismo_aeropuerto = origen == destino
        if mismo_aeropuerto:
            st.error("⚠️ El origen y el destino no pueden ser el mismo.")

        # Validación 2: Filtro de aerolíneas por ruta
        es_usa = "JFK" in origen or "JFK" in destino
        es_europa = "MAD" in origen or "MAD" in destino
        
        if es_europa:
            opciones_aero = rutas_operativas["Transatlántico (Europa)"]
        elif es_usa:
            opciones_aero = rutas_operativas["Internacional (USA)"]
        else:
            opciones_aero = rutas_operativas["Nacional (México)"]

        aerolinea = st.selectbox("🏢 Aerolínea disponible", opciones_aero)
        fecha = st.date_input("📅 Fecha de salida", value=date.today())
        hora = st.slider("🕒 Hora de salida", 0, 23, 12)
        
        # Botón habilitado solo si la ruta es válida
        btn_analizar = st.button(
            "🚀 ANALIZAR RETRASO", 
            use_container_width=True, 
            disabled=mismo_aeropuerto
        )

    if btn_analizar:
        if modelo_reg:
            with st.spinner('Analizando condiciones en tiempo real...'):
                clima = obtener_clima_real(origen, fecha)
                repu = reputacion_dict.get(aerolinea, 0.80)
                
                # Features: [fase_dia, reputation, flights, visibility, Hora, temp, wind, dia_semana, precip]
                features = [
                    1 if 6 <= hora <= 18 else 0, 
                    repu, 25, clima['vis'], int(hora), 
                    clima['temp'], clima['wind'], int(fecha.weekday()), clima['precip']
                ]
                
                # Predicción y ajuste de escala (Sensibilidad)
                datos_array = np.array([features], dtype=float)
                pred_raw = modelo_reg.predict(datos_array)[0]
                
                # Compensación para el modelo de 30 días
                minutos = pred_raw
                if (clima['precip'] > 3 or clima['vis'] < 8) and minutos < 10:
                    minutos = minutos * 10
                
                minutos_final = int(max(0, round(minutos)))

                st.session_state.resultado_final = {
                    'minutos': minutos_final, 'aero': aerolinea, 
                    'clima': clima, 'ruta': f"{origen} a {destino}"
                }

    # Resultados del Análisis
    if st.session_state.resultado_final:
        res = st.session_state.resultado_final
        st.markdown("---")
        st.write("### 🛡️ Diagnóstico de Vuelo")
        c1, c2 = st.columns(2)
        c1.metric("Retraso Estimado", f"{res['minutos']} min")
        c2.metric("Clima Detectado", f"{res['clima']['temp']}°C")
        
        if res['minutos'] > 30:
            st.error(f"Se prevé un retraso considerable con {res['aero']}.")
        else:
            st.success(f"Todo indica que el vuelo de {res['ruta']} será puntual.")

# --- COLUMNA DERECHA: CHAT CONECTIA ---
with col_chat:
    st.subheader("🤖 Asistente ConectIA")
    
    # Altura fija para mantener el formulario siempre a la vista
    chat_box = st.container(height=520, border=True)
    
    with chat_box:
        if not st.session_state.messages:
            st.info("¡Hola! Soy ConectIA. Analiza tu vuelo a la izquierda y te daré consejos personalizados.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Pregúntame sobre el clima o tu retraso..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                # Contexto dinámico
                ctx = ""
                if st.session_state.resultado_final:
                    r = st.session_state.resultado_final
                    ctx = f" El usuario analizó un vuelo de {r['aero']} con {r['minutos']} min de retraso por clima de {r['clima']['temp']} grados."
                
                try:
                    full_query = f"Eres ConectIA, un asistente de viajes amable y servicial.{ctx} Responde: {prompt}"
                    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_query}])
                    st.markdown(response.choices[0].message.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
                except:
                    st.error("Error al conectar con la IA.")
        st.rerun()

st.markdown("---")
st.caption("ConectIA v2.5 | Seguridad y Puntualidad en tus Manos")



