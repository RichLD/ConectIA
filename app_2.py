import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
# Asegúrate de tener estos nombres en tus Secrets de Streamlit
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
AVIATION_API_KEY = st.secrets["AVIATION_API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def cargar_modelos():
    modelo_clas = None
    modelo_reg = None
    
    # Cargar Clasificador (Riesgo %)
    paths_clas = ['modelo_conectia.pkl', 'models/modelo_conectia.pkl']
    for path in paths_clas:
        if os.path.exists(path):
            modelo_clas = joblib.load(path)
            break
            
    # Cargar Regresor (Tiempo min) - Formato JSON de XGBoost
    path_reg = 'modelo_vuelos_regresion.json'
    if os.path.exists(path_reg):
        modelo_reg = XGBRegressor()
        modelo_reg.load_model(path_reg)
        
    return modelo_clas, modelo_reg

modelo_xgb, modelo_reg = cargar_modelos()

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
    if st.button("💬 Chat de Ayuda", use_container_width=True): st.session_state.etapa = 'chat'
    if st.button("📊 Revisar mi Vuelo", use_container_width=True): st.session_state.etapa = 'formulario'

# --- 6. PANTALLA: CHAT ---
if st.session_state.etapa == 'chat':
    st.header("🤖 ¿Cómo podemos ayudarte hoy?")
    # (Lógica de chat omitida por brevedad, se mantiene igual a tu código original)
    with st.chat_message("assistant"):
        st.write("### **¡Hola! Qué gusto saludarte.** 😊")
        st.write("Soy ConectIA, tu asistente personal. ¿En qué puedo ayudarte?")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aquí tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            contexto = f" El usuario tiene un riesgo de {st.session_state.resultado_final['prob']*100:.1f}%." if 'resultado_final' in st.session_state else ""
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": f"Eres ConectIA, asistente de viajes.{contexto} Responde: {prompt}"}]
                )
                res_text = response.choices[0].message.content
                st.markdown(res_text)
                st.session_state.messages.append({"role": "assistant", "content": res_text})
            except: st.error("Error al procesar.")

# --- 7. PANTALLA: FORMULARIO ---
elif st.session_state.etapa == 'formulario':
    st.header("📊 Analizador de Riesgo y Tiempo")
    tab_api, tab_manual = st.tabs(["🔍 Buscar por Número de Vuelo", "⚙️ Configurar Manualmente"])
    disparar_prediccion = False

    with tab_api:
        c_search, c_btn = st.columns([3, 1])
        v_in = c_search.text_input("Número de vuelo", placeholder="Ej: VB1024")
        if c_btn.button("Buscar Vuelo"):
            v_data = buscar_vuelo_tiempo_real(v_in)
            if v_data:
                st.session_state.origen_v, st.session_state.aerolinea_v = v_data['origen_iata'], v_data['aerolinea'].replace(" ", "")
                st.success("✅ Vuelo encontrado."); disparar_prediccion = True
            else: st.error("No encontrado.")

    with tab_manual:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                idx_o = next((i for i, s in enumerate(aeropuertos) if st.session_state.get('origen_v', '') in s), 0)
                origen = st.selectbox("📍 ¿De dónde sales?", aeropuertos, index=idx_o)
                destino = st.selectbox("🏁 ¿A dónde vas?", aeropuertos, index=1)
                
                opciones_aero = rutas_operativas["Nacional (México)"]
                if "MAD" in origen or "MAD" in destino: opciones_aero = rutas_operativas["Transatlántico (Europa)"]
                elif "JFK" in origen or "JFK" in destino: opciones_aero = rutas_operativas["Internacional (USA)"]
                
                aero_api = st.session_state.get('aerolinea_v', opciones_aero[0])
                idx_a = opciones_aero.index(aero_api) if aero_api in opciones_aero else 0
                aerolinea = st.selectbox("🏢 Aerolínea", opciones_aero, index=idx_a)
            
            with col2:
                fecha = st.date_input("📅 ¿Qué día viajas?", value=date.today())
                hora = st.slider("🕒 ¿A qué hora sale tu vuelo?", 0, 23, 12)

        if st.button("🚀 CALCULAR PREDICCIÓN", use_container_width=True): disparar_prediccion = True

    # --- 8. PROCESAMIENTO DUAL (Clasificación + Regresión) ---
    if disparar_prediccion:
        if origen == destino: st.error("❌ Origen y destino iguales.")
        elif modelo_xgb and modelo_reg:
            with st.spinner('Analizando condiciones climatológicas y tráfico...'):
                clima = obtener_clima_inteligente(origen, fecha)
                p, w, v, temp = float(clima['precip']), float(clima['wind']), float(clima['vis']), float(clima['temp'])
                repu = float(reputacion_dict.get(aerolinea, 0.80))
                
                # Datos preparados para ambos modelos (9 variables)
                datos = pd.DataFrame([[
                    temp, p, w, v, 
                    1 if 6 <= hora <= 18 else 0, # fase_dia
                    int(fecha.weekday()), 
                    int(hora), 
                    25, # flights_at_that_hour (estático o promedio)
                    repu
                ]], columns=['temp', 'precip', 'windspeed', 'visibility', 
                            'fase_dia', 'dia_semana', 'Hora', 
                            'flights_at_that_hour', 'airline_reputation_score'])

                # Predicción 1: Probabilidad de Retraso
                prob_raw = modelo_xgb.predict_proba(datos)[0][1]
                prob_final = max(0.05, min((prob_raw * 0.85) + (p * 0.0015), 0.95))

                # Predicción 2: Tiempo en Minutos
                minutos_raw = modelo_reg.predict(datos)[0]
                minutos_final = max(0, minutos_raw)

                try:
                    res_ia = client.chat.completions.create(
                        model="gpt-3.5-turbo", 
                        messages=[{"role": "user", "content": f"Vuelo {aerolinea}, riesgo {prob_final*100:.1f}%, tiempo {minutos_final:.0f} min. Dame un consejo breve."}]
                    )
                    consejo = res_ia.choices[0].message.content
                except: consejo = "Llega con anticipación al aeropuerto."

                st.session_state.resultado_final = {
                    'prob': prob_final, 
                    'minutos': minutos_final, 
                    'aero': aerolinea, 
                    'ruta': f"{origen} ➔ {destino}", 
                    'consejo': consejo
                }

    # --- 9. DESPLIEGUE DE RESULTADOS ---
    if 'resultado_final' in st.session_state:
        res = st.session_state.resultado_final
        st.divider()
        st.subheader(f"🛡️ Diagnóstico ConectIA para {res['aero']}")
        
        col_met1, col_met2, col_txt = st.columns([1, 1, 2])
        col_met1.metric("Probabilidad de Retraso", f"{res['prob']*100:.1f}%")
        col_met2.metric("Retraso Estimado", f"{int(res['minutos'])} min")
        
        if res['prob'] > 0.60 or res['minutos'] > 45:
            col_txt.error(f"**Alerta:** Alta probabilidad de demoras importantes en {res['ruta']}.")
        elif res['prob'] > 0.30 or res['minutos'] > 15:
            col_txt.warning(f"**Aviso:** Podrías tener un retraso moderado en {res['ruta']}.")
        else:
            col_txt.success(f"**Despejado:** Todo indica que tu vuelo en {res['ruta']} será puntual.")
            
        st.info(f"**💡 Consejo de ConectIA:** {res['consejo']}")
