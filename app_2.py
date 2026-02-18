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
    # El archivo debe estar en la raíz de tu repo de GitHub
    path = 'modelo_vuelos_regresion.json'
    if os.path.exists(path):
        model = XGBRegressor()
        model.load_model(path)
        return model
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
        if 'days' in res:
            dia = res['days'][0]
            return {
                'temp': dia.get('temp', 22.0), 
                'precip': dia.get('precip', 0.0), 
                'wind': dia.get('windspeed', 12.0), 
                'vis': dia.get('visibility', 15.0)
            }
    except: 
        return {'temp': 22.0, 'precip': 0.0, 'wind': 12.0, 'vis': 15.0}

def buscar_vuelo_tiempo_real(flight_iata):
    url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATION_API_KEY}&flight_iata={flight_iata}"
    try:
        res = requests.get(url, timeout=10).json()
        if res.get('data'):
            vuelo = res['data'][0]
            return {'origen_iata': vuelo['departure']['iata'], 'aerolinea': vuelo['airline']['name']}
    except: 
        return None

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
        st.write("Soy ConectIA, tu asistente personal de viajes. Pregúntame sobre tus vuelos o consejos para tu travesía.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): 
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aquí tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            contexto_vuelo = ""
            if 'resultado_final' in st.session_state:
                res = st.session_state.resultado_final
                contexto_vuelo = f" El usuario tiene un vuelo con {res['aero']} de {res['ruta']} y el retraso estimado es de {res['minutos']} minutos."
            
            try:
                # Mantenemos el prompt de personalidad solicitado
                full_query = f"Eres ConectIA, un asistente de viajes amable y servicial.{contexto_vuelo} Responde: {prompt}"
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_query}])
                respuesta = response.choices[0].message.content
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except: 
                st.error("Lo siento, tuve un problema al procesar tu duda.")

# --- 7. PANTALLA: FORMULARIO ---
elif st.session_state.etapa == 'formulario':
    st.header("📊 Analizador de Retraso")
    tab_api, tab_manual = st.tabs(["🔍 Buscar por Número de Vuelo", "⚙️ Configurar Manualmente"])
    disparar_prediccion = False

    with tab_api:
        c_search, c_btn = st.columns([3, 1])
        v_in = c_search.text_input("Número de vuelo", placeholder="Ej: AM240", key="flight_input")
        if c_btn.button("Buscar Vuelo"):
            v_data = buscar_vuelo_tiempo_real(v_in)
            if v_data:
                st.session_state.origen_v = v_data['origen_iata']
                st.session_state.aerolinea_v = v_data['aerolinea'].replace(" ", "")
                st.success("✅ Vuelo encontrado.")
                disparar_prediccion = True
            else: 
                st.error("No se encontró el vuelo.")

    with tab_manual:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                idx_o = next((i for i, s in enumerate(aeropuertos) if st.session_state.get('origen_v', '') in s), 0)
                origen = st.selectbox("📍 ¿De dónde sales?", aeropuertos, index=idx_o)
                destino = st.selectbox("🏁 ¿A dónde vas?", aeropuertos, index=1)
                
                if "MAD" in origen or "MAD" in destino: 
                    opciones_aero = rutas_operativas["Transatlántico (Europa)"]
                elif "JFK" in origen or "JFK" in destino: 
                    opciones_aero = rutas_operativas["Internacional (USA)"]
                else: 
                    opciones_aero = rutas_operativas["Nacional (México)"]
                
                aero_api = st.session_state.get('aerolinea_v', opciones_aero[0])
                idx_a = opciones_aero.index(aero_api) if aero_api in opciones_aero else 0
                aerolinea = st.selectbox("🏢 Aerolínea", opciones_aero, index=idx_a)
            
            with col2:
                fecha = st.date_input("📅 ¿Qué día viajas?", value=date.today())
                hora = st.slider("🕒 ¿A qué hora sale tu vuelo?", 0, 23, 12)

        if st.button("🚀 ESTIMAR TIEMPO", use_container_width=True): 
            disparar_prediccion = True

    # --- 8. PROCESAMIENTO ---
    if disparar_prediccion:
        if origen == destino: 
            st.error("❌ El origen y destino no pueden ser iguales.")
        elif modelo_reg:
            with st.spinner('Analizando condiciones climatológicas y tráfico...'):
                clima = obtener_clima_inteligente(origen, fecha)
                p, w, v, temp = float(clima['precip']), float(clima['wind']), float(clima['vis']), float(clima['temp'])
                repu = float(reputacion_dict.get(aerolinea, 0.80))
                
                # ORDEN ESTRICTO DE COLUMNAS (Basado en el entrenamiento del regresor)
                orden_columnas = [
                    'fase_dia', 'airline_reputation_score', 'flights_at_that_hour', 
                    'visibility', 'Hora', 'temp', 'windspeed', 'dia_semana', 'precip'
                ]
                
                input_data = pd.DataFrame([{
                    'fase_dia': 1 if 6 <= hora <= 18 else 0,
                    'airline_reputation_score': repu,
                    'flights_at_that_hour': 25,
                    'visibility': v,
                    'Hora': int(hora),
                    'temp': temp,
                    'windspeed': w,
                    'dia_semana': int(fecha.weekday()),
                    'precip': p
                }])[orden_columnas]

                # Predicción usando .values para evitar el error de feature_names
                minutos = int(max(0, modelo_reg.predict(input_data.values)[0]))

                # Lógica de la IA para el consejo (Prompt original mantenido)
                try:
                    full_query_ia = (f"Eres ConectIA, un asistente de viajes amable y servicial. "
                                     f"El usuario tiene un vuelo con {aerolinea} de {origen} a {destino} "
                                     f"y el retraso estimado es de {minutos} minutos. Responde con un consejo breve.")
                    
                    response_ia = client.chat.completions.create(
                        model="gpt-3.5-turbo", 
                        messages=[{"role": "user", "content": full_query_ia}]
                    )
                    consejo = response_ia.choices[0].message.content
                except:
                    consejo = "Te recomendamos revisar el estatus de tu vuelo antes de salir al aeropuerto."

                st.session_state.resultado_final = {
                    'minutos': minutos, 
                    'aero': aerolinea, 
                    'ruta': f"{origen} ➔ {destino}", 
                    'consejo': consejo
                }

    # --- 9. DESPLIEGUE GLOBAL ---
    if 'resultado_final' in st.session_state:
        res = st.session_state.resultado_final
        st.divider()
        st.subheader(f"🛡️ Diagnóstico de ConectIA: {res['aero']}")
        c1, c2 = st.columns([1, 2])
        c1.metric("Retraso Estimado", f"{res['minutos']} min")
        
        if res['minutos'] > 45: 
            c2.error(f"**Atención:** Se estima un retraso importante de {res['minutos']} minutos.")
        elif res['minutos'] > 15: 
            c2.warning(f"**Aviso:** Podrías tener una demora leve en tu itinerario.")
        else: 
            c2.success(f"**Todo en orden:** El modelo estima que tu vuelo será puntual.")
        
        st.info(f"**💡 Consejo de ConectIA:** {res['consejo']}")

