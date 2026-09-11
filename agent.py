import logging
import os
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Cargar variables de entorno
load_dotenv()

# --- LOGGING TEMPORAL: registra cada llamada REST y cada llamada al modelo ---
# TODO: quitar este bloque, log_rest() y su uso en las tools y en create_agent().

# langchain_mistralai usa httpx por debajo: su logger INFO imprime cada POST a api.mistral.ai
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.INFO)

_CLAVES_SENSIBLES = {"key", "api_key", "apikey", "token"}

def log_rest(metodo: str, url: str, params: dict, resp: requests.Response) -> None:
    """Imprime una llamada REST hecha con requests, ocultando credenciales de la query."""
    seguros = {
        k: ("***" if k.lower() in _CLAVES_SENSIBLES else v)
        for k, v in (params or {}).items()
    }
    print(
        f"[🌐 REST] {metodo} {url}?{urlencode(seguros)} "
        f"-> {resp.status_code} ({resp.elapsed.total_seconds():.2f}s)"
    )

class LoggingCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"\n[🤖 MISTRAL] --> Llamada al modelo con {len(messages[0])} mensaje(s) de contexto:")
        for m in messages[0]:
            contenido = str(m.content)[:120].replace("\n", " ")
            print(f"    [{m.type}] {contenido}")

    def on_llm_end(self, response, **kwargs):
        mensaje = response.generations[0][0].message
        tool_calls = getattr(mensaje, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"[🤖 MISTRAL] <-- Pide ejecutar tool: {tc['name']}({tc['args']})")
        else:
            print(f"[🤖 MISTRAL] <-- Respuesta final: {str(mensaje.content)[:200]}")

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not WEATHER_API_KEY or not MISTRAL_API_KEY:
    raise ValueError("❌ ERROR CRÍTICO: Faltan credenciales. Asegúrate de que el archivo .env existe y contiene WEATHER_API_KEY y MISTRAL_API_KEY.")

# --- HERRAMIENTA 1: CLIMA ---
@tool
def consultar_clima(ciudad: str) -> str:
    """Consulta el clima actual de una ciudad. Útil para saber si hace sol, llueve o hace frío."""
    print(f"\n[🌤️  CLIMA] Consultando clima en {ciudad}...")
    url = "http://api.weatherapi.com/v1/current.json"
    params = {"key": WEATHER_API_KEY, "q": ciudad, "lang": "es"}
    
    try:
        resp = requests.get(url, params=params)
        log_rest("GET", url, params, resp)  # TEMPORAL
        resp.raise_for_status()
        datos = resp.json()
        
        temp = datos["current"]["temp_c"]
        cond = datos["current"]["condition"]["text"]
        return f"En {ciudad} hace {temp}°C y está {cond}."
    except Exception as e:
        return f"Error de conexión con WeatherAPI: {str(e)}"

# --- HERRAMIENTA 2: MÚSICA (Vía iTunes) ---
@tool
def buscar_musica_por_animo(animo: str) -> str:
    """
    Busca una canción recomendada basada en un estado de ánimo o clima.
    Ejemplo de entrada: 'lluvia', 'sol', 'fiesta', 'música tranquila'.
    """
    print(f"[🎵 MÚSICA] Buscando recomendaciones para: '{animo}'...")
    url = "https://itunes.apple.com/search"
    params = {"term": animo, "media": "music", "entity": "song", "limit": 1}
    
    try:
        resp = requests.get(url, params=params)
        log_rest("GET", url, params, resp)  # TEMPORAL
        resp.raise_for_status()
        datos = resp.json()

        if datos["resultCount"] == 0:
            return "No encontré canciones para ese estado de ánimo."
            
        cancion = datos["results"][0]
        titulo = cancion["trackName"]
        artista = cancion["artistName"]
        url_link = cancion["trackViewUrl"]
        
        return f"Te recomiendo escuchar '{titulo}' de {artista}. Enlace: {url_link}"
    except Exception as e:
        return f"Error al buscar música: {str(e)}"

# Modelo por defecto si el cliente no indica ninguno en el POST /chat
MODELO_POR_DEFECTO = os.getenv("MISTRAL_MODEL", "ministral-3b-latest")

# Checkpointer compartido: así la memoria de una sesión (thread_id) persiste
# aunque dos mensajes de la misma conversación usen modelos distintos.
_memory = MemorySaver()

# --- CONSTRUCTOR DEL AGENTE ---
def create_agent(model: str = MODELO_POR_DEFECTO):
    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=model,
        temperature=0.0,
        callbacks=[LoggingCallbackHandler()],  # TEMPORAL: quitar cuando termines de debuggear
    )

    herramientas = [consultar_clima, buscar_musica_por_animo]

    instrucciones = (
        "Eres un asistente conversacional útil y experto.\n"
        "REGLA 1: Si el usuario pregunta por el clima, usa la herramienta de clima.\n"
        "REGLA 2: Si pide música basada en el clima, usa primero el clima y luego busca música.\n"
        "REGLA 3: Redacta siempre una respuesta final amigable para el usuario con los datos obtenidos."
    )

    # Creamos el grafo del agente inyectando la memoria y el prompt del sistema
    agent_graph = create_react_agent(
        model=llm,           # <-- (Opcional) En versiones nuevas, el parámetro se llama formalmente 'model'
        tools=herramientas,
        checkpointer=_memory,
        prompt=instrucciones # <-- ¡EL CAMBIO ESTÁ AQUÍ! Reemplazamos state_modifier por prompt
    )

    return agent_graph