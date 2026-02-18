import streamlit as st
import pandas as pd
import joblib
import os
import requests
from openai import OpenAI
from datetime import datetime, date

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="ConectIA - Tu Asistente de Viajes",
    page_icon="✈️",
    layout="wide"
)

# --- 2. CONFIGURACIÓN DE CLIENTES ---

openai.api_key = st.secrets["OPENAI_API_KEY"]
weather_api_key = st.secrets["WEATHER_API_KEY"]
aviation_api_key = st.secrets["AVIATION_API_KEY"]
client = OpenAI(api_key=OPENAI_KEY)

@st.cache_resource
def cargar_modelo():
    paths = ['modelo_conectia.pkl', 'models/modelo_conectia.pkl']
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

modelo_xgb = cargar_modelo()

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
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{ciudad_query}/{fecha_str}?unitGroup=metric&key={WEATHER_KEY}&contentType=json&include=days"
    try:
        res = requests.get(url, timeout=10).json()
        if 'days' in res:
            dia = res['days'][0]
            return {'temp': dia.get('temp', 22.0), 'precip': dia.get('precip', 0.0), 'wind': dia.get('windspeed', 12.0), 'vis': dia.get('visibility', 15.0)}
    except: return {'temp': 22.0, 'precip': 0.0, 'wind': 12.0, 'vis': 15.0}

def buscar_vuelo_tiempo_real(flight_iata):
    url = f"http://api.aviationstack.com/v1/flights?access_key={AVIATIONSTACK_KEY}&flight_iata={flight_iata}"
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
    st.markdown("---")
    if st.button("💬 Chat de Ayuda", use_container_width=True): st.session_state.etapa = 'chat'
    if st.button("📊 Revisar mi Vuelo", use_container_width=True): st.session_state.etapa = 'formulario'

# --- 6. PANTALLA: CHAT ---
if st.session_state.etapa == 'chat':
    st.header("🤖 ¿Cómo podemos ayudarte hoy?")
    
    with st.chat_message("assistant"):
        st.write("### **¡Hola! Qué gusto saludarte.** 😊")
        st.write("""
        Soy ConectIA, tu asistente personal para que tu viaje sea lo más tranquilo posible. 
        
        **¿Qué puedo hacer por ti?**
        * **Aclarar dudas:** Si ya revisaste tu vuelo, puedo explicarte qué significa el nivel de riesgo que salió.
        * **Darte consejos:** Pregúntame qué hacer si hay mal clima o cómo prepararte para tu conexión.
        * **Ayudarte con el plan:** Dime a dónde vas y te daré los mejores tips para evitar retrasos.
        
        *Cuéntame, ¿tienes alguna duda con tu vuelo o quieres que planeemos algo juntos?*
        """)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aquí tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            contexto_adicional = ""
            if 'resultado_final' in st.session_state:
                res = st.session_state.resultado_final
                contexto_adicional = f" El usuario tiene un vuelo con {res['aero']} de {res['ruta']} y el riesgo es del {res['prob']*100:.1f}%."
            try:
                full_query = f"Eres ConectIA, un asistente de viajes amable y servicial.{contexto_adicional} Responde: {prompt}"
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": full_query}])
                respuesta = response.choices[0].message.content
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
            except: st.error("Lo siento, tuve un pequeño problema al procesar tu duda.")

# --- 7. PANTALLA: FORMULARIO ---
elif st.session_state.etapa == 'formulario':
    with st.chat_message("assistant"):
        st.write("### **¡Vamos a revisar tu vuelo!** ✈️")
        st.write("""
        Para decirte qué tan probable es que tu vuelo se retrase, necesito un par de datos.
        
        Puedes **escribir tu número de vuelo** para que yo busque la información automáticamente, o puedes **llenar los datos tú mismo** si prefieres planear con anticipación.
        """)

    st.header("📊 Analizador de Riesgo")
    tab_api, tab_manual = st.tabs(["🔍 Buscar por Número de Vuelo", "⚙️ Configurar Manualmente"])
    disparar_prediccion = False

    with tab_api:
        c_search, c_btn = st.columns([3, 1])
        v_in = c_search.text_input("Número de vuelo", placeholder="Ej: VB1024")
        if c_btn.button("Buscar Vuelo"):
            v_data = buscar_vuelo_tiempo_real(v_in)
            if v_data:
                st.session_state.origen_v, st.session_state.aerolinea_v = v_data['origen_iata'], v_data['aerolinea'].replace(" ", "")
                st.success("✅ Vuelo encontrado. Ya tengo la información lista."); disparar_prediccion = True
            else: st.error("No pude encontrar ese vuelo. Revisa el número o intenta la opción manual.")

    with tab_manual:
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                idx_o = next((i for i, s in enumerate(aeropuertos) if st.session_state.get('origen_v', '') in s), 0)
                origen = st.selectbox("📍 ¿De dónde sales?", aeropuertos, index=idx_o)
                destino = st.selectbox("🏁 ¿A dónde vas?", aeropuertos, index=1)
                
                if "MAD" in origen or "MAD" in destino: opciones_aero = rutas_operativas["Transatlántico (Europa)"]
                elif "JFK" in origen or "JFK" in destino: opciones_aero = rutas_operativas["Internacional (USA)"]
                else: opciones_aero = rutas_operativas["Nacional (México)"]
                
                aero_api = st.session_state.get('aerolinea_v', opciones_aero[0])
                idx_a = opciones_aero.index(aero_api) if aero_api in opciones_aero else 0
                aerolinea = st.selectbox("🏢 Aerolínea", opciones_aero, index=idx_a)
            
            with col2:
                fecha, hora = st.date_input("📅 ¿Qué día viajas?", value=date.today()), st.slider("🕒 ¿A qué hora sale tu vuelo?", 0, 23, 12)
                notas = st.text_area("📝 ¿Algún detalle extra?")

        if st.button("🚀 CALCULAR RIESGO", use_container_width=True): disparar_prediccion = True

    # --- 8. PROCESAMIENTO ---
    if disparar_prediccion:
        if origen == destino: 
            st.error("❌ Conflicto de ruta: Origen y destino idénticos.")
        elif modelo_xgb:
            with st.spinner('Analizando condiciones...'):
                clima = obtener_clima_inteligente(origen, fecha)
                p, w, v, temp = float(clima['precip']), float(clima['wind']), float(clima['vis']), float(clima['temp'])
                repu = float(reputacion_dict[aerolinea])
                
                # CREACIÓN DE DATOS (Solo las 9 variables que tu modelo reconoce)
                datos = pd.DataFrame([[
                    temp, 
                    p, 
                    w, 
                    v, 
                    1 if 6 <= hora <= 18 else 0, # fase_dia
                    int(fecha.weekday()),        # dia_semana
                    int(hora),                   # Hora
                    25,                          # flights_at_that_hour
                    repu                         # airline_reputation_score
                ]], columns=[
                    'temp', 'precip', 'windspeed', 'visibility', 
                    'fase_dia', 'dia_semana', 'Hora', 
                    'flights_at_that_hour', 'airline_reputation_score'
                ])

                # Ahora sí, la predicción no fallará
                prob_raw = modelo_xgb.predict_proba(datos)[0][1]
                
                # Calibración final
                f_clima = (p * 0.0015) + (w * 0.0001) 
                f_repu = (0.90 - repu) * 0.05
                
                prob_final = (prob_raw * 0.85) + f_clima + f_repu
                prob_final = max(0.05, min(prob_final, 0.85))
                try:
                    res_ia = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Vuelo {aerolinea}, riesgo {prob_final*100:.1f}%. Da un consejo de viaje amable de 2 líneas."}])
                    consejo = res_ia.choices[0].message.content
                except: consejo = "Te recomendamos llegar con tiempo al aeropuerto."

                st.session_state.resultado_final = {'prob': prob_final, 'raw': prob_raw, 'aero': aerolinea, 'ruta': f"{origen} ➔ {destino}", 'consejo': consejo}

    # --- 9. DESPLIEGUE GLOBAL ---
    if 'resultado_final' in st.session_state:
        res = st.session_state.resultado_final
        st.divider()
        st.subheader(f"🛡️ Tu Diagnóstico de Viaje: {res['aero']}")
        c1, c2 = st.columns([1, 2])
        c1.metric("Posibilidad de Retraso", f"{res['prob']*100:.1f}%")
        
        if res['prob'] > 0.65: c2.error(f"**Ten cuidado:** Parece que hay alta probabilidad de demoras para {res['ruta']}.")
        elif res['prob'] > 0.35: c2.warning(f"**Toma precauciones:** Podría haber algunos retrasos ligeros en {res['ruta']}.")
        else: c2.success(f"**¡Todo bien!:** Tu vuelo en la ruta {res['ruta']} se ve muy estable.")
        
        st.info(f"**💡 Mi consejo para ti:** {res['consejo']}")