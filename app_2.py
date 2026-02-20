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

# --- 3. DICCIONARIOS Y LOGÍSTICA ---
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

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'resultado_final' not in st.session_state:
    st.session_state.resultado_final = None

# --- 4. FUNCIÓN CLIMA REAL (CON PLAN B POR CRÉDITOS) ---
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

# --- 5. INTERFAZ UNIFICADA ---
st.title("✈️ ConectIA: Simulador de Vuelos Inteligente")
st.markdown("---")

col_form, col_chat = st.columns([1, 1], gap="large")

# --- COLUMNA IZQUIERDA: FORMULARIO ---
with col_form:
    st.subheader("📊 Configuración del Vuelo")
    with st.container(border=True):
        # Destinos
        c1, c2 = st.columns(2)
        with c1:
            origen = st.selectbox("📍 Ciudad de Origen", aeropuertos)
        with c2:
            destino = st.selectbox("🏁 Ciudad de Destino", aeropuertos, index=1)
        
        # VALIDACIÓN: No viajar al mismo aeropuerto
        mismo_aeropuerto = (origen == destino)
        if mismo_aeropuerto:
            st.error("⚠️ El origen y el destino no pueden ser iguales.")

        # FILTRO DE AEROLÍNEAS (Regla Nacional vs Internacional)
        es_usa = "JFK" in origen or "JFK" in destino
        es_europa = "MAD" in origen or "MAD" in destino
        
        if es_europa:
            opciones_aero = rutas_operativas["Europa"]
        elif es_usa:
            opciones_aero = rutas_operativas["USA"]
        else:
            opciones_aero = rutas_operativas["Nacional"] # Aquí American Airlines NO aparece

        aerolinea = st.selectbox("🏢 Aerolínea disponible", opciones_aero)
        
        # Fecha y Hora (Funciones recuperadas)
        c3, c4 = st.columns(2)
        with c3:
            fecha = st.date_input("📅 Fecha de salida", value=date.today())
        with c4:
            hora = st.slider("🕒 Hora de salida", 0, 23, 12)
        
        # Botón de análisis
        btn_analizar = st.button("🚀 REALIZAR ANÁLISIS", use_container_width=True, disabled=mismo_aeropuerto)

    if btn_analizar and modelo_reg:
        with st.spinner('Procesando datos del clima y vuelo...'):
            clima = obtener_clima_real(origen, fecha)
            
            # Si no hay créditos, forzamos clima que genere impacto para no ver 0 siempre
            if clima['status'] == 'error':
                st.warning("⚠️ Usando estimación climática (API sin créditos).")
                clima['precip'] = 7.5 
                clima['vis'] = 5.0
                clima['wind'] = 25.0
            
            repu = reputacion_dict.get(aerolinea, 0.80)
            
            # Vector de características para XGBoost
            features = [
                1 if 6 <= hora <= 18 else 0, 
                repu, 25, clima['vis'], int(hora), 
                clima['temp'], clima['wind'], int(fecha.weekday()), clima['precip']
            ]
            
            # Predicción
            datos_array = np.array([features], dtype=float)
            pred_raw = modelo_reg.predict(datos_array)[0]
            
            # Ajuste de sensibilidad
            minutos = pred_raw
            if (clima['precip'] > 2 or clima['vis'] < 8) and minutos < 10:
                minutos = minutos * 10
            
            minutos_final = int(max(0, round(minutos)))

            st.session_state.resultado_final = {
                'minutos': minutos_final, 'aero': aerolinea, 
                'clima': clima, 'ruta': f"{origen} a {destino}"
            }

    # Despliegue de resultados debajo del botón
    if st.session_state.resultado_final:
        res = st.session_state.resultado_final
        st.markdown("---")
        st.write("### 🛡️ Resultado del Análisis")
        c_res1, c_res2, c_res3 = st.columns(3)
        c_res1.metric("Retraso", f"{res['minutos']} min")
        c_res2.metric("Temp.", f"{res['clima']['temp']}°C")
        c_res3.metric("Lluvia", f"{res['clima']['precip']} mm")
        
        if res['minutos'] > 30:
            st.error(f"Riesgo de demora importante con {res['aero']}.")
        else:
            st.success(f"Vuelo puntual detectado para {res['ruta']}.")

# --- COLUMNA DERECHA: CHAT CONECTIA ---
with col_chat:
    st.subheader("🤖 Chat ConectIA")

    chat_box = st.container(border=True)

    with chat_box:
        if not st.session_state.get("messages"):
            st.info("""
👋 **Bienvenido a ConectIA**

Soy tu asistente inteligente de vuelos ✈️  

🔎 Analizo el retraso de tu vuelo  
⚠️ Evalúo el nivel de impacto en tu logística  
✅ Te doy acciones concretas para minimizar problemas  

Puedes preguntarme cosas como:
- ¿Voy a perder mi conexión?
- ¿Debo cambiar mi transporte?
- ¿Aplica compensación?
- ¿Qué hago si el retraso aumenta?

Analiza tu vuelo a la izquierda y luego hazme tu pregunta aquí 👇
""")

        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Escribe tu duda..."):

        # Guardar mensaje usuario
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):

                # ---------------------------
                # CONTEXTO DEL VUELO
                # ---------------------------
                r = st.session_state.get("resultado_final")
                ctx = ""
                impacto = "No determinado"

                if r:
                    minutos = r.get("minutos", 0)

                    if minutos <= 15:
                        impacto = "Bajo"
                    elif minutos <= 45:
                        impacto = "Moderado"
                    elif minutos <= 120:
                        impacto = "Alto"
                    else:
                        impacto = "Crítico"

                    ctx = f"""
Contexto del vuelo:
- Aerolínea/Aeropuerto: {r.get('aero', 'No especificado')}
- Retraso actual: {minutos} minutos
- Nivel estimado de impacto: {impacto}
"""

                # ---------------------------
                # PROMPT SISTEMA
                # ---------------------------
                system_prompt = """
Eres ConectIA, asistente experto en vuelos y logística de pasajeros.

Tu objetivo es minimizar el impacto del retraso en la logística del pasajero.

Responde SIEMPRE en formato JSON con esta estructura:

{
  "diagnostico": "",
  "nivel_impacto": "",
  "acciones_recomendadas": [],
  "consejo_adicional": ""
}
"""

                user_prompt = f"{ctx}\nPregunta del pasajero:\n{prompt}"

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.4,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )

                    import json
                    data = json.loads(response.choices[0].message.content)

                    # Render estructurado
                    st.markdown("### 📊 Diagnóstico")
                    st.write(data["diagnostico"])

                    st.markdown("### ⚠️ Nivel de Impacto")
                    st.write(data["nivel_impacto"])

                    st.markdown("### ✅ Acciones Recomendadas")
                    for accion in data["acciones_recomendadas"]:
                        st.write(f"- {accion}")

                    st.markdown("### 💡 Consejo Adicional")
                    st.write(data["consejo_adicional"])

                    # Guardar versión formateada en memoria
                    formatted_response = f"""
📊 **Diagnóstico:**  
{data["diagnostico"]}

⚠️ **Nivel de Impacto:**  
{data["nivel_impacto"]}

✅ **Acciones Recomendadas:**  
{chr(10).join(['- ' + a for a in data["acciones_recomendadas"]])}

💡 **Consejo Adicional:**  
{data["consejo_adicional"]}
"""

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": formatted_response
                    })

                except Exception as e:
                    st.error("Error al conectar con ConectIA.")
                    st.exception(e)

        st.rerun()

#FINAL


