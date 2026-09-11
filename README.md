# Weather & Music AI Agent

Este es un agente conversacional inteligente construido con FastAPI, LangChain y LangGraph. Utiliza el modelo de **Mistral AI** para procesar lenguaje natural y tiene acceso a herramientas para consultar el clima actual y buscar recomendaciones musicales basadas en el estado de ánimo o el clima.

## Configuración de Variables de Entorno

Para que el proyecto funcione correctamente, es **estrictamente necesario** configurar las siguientes dos variables de entorno. Al desplegar la aplicación, asegúrate de añadirlas en la sección de **Environment Variables (Optional)** de tu plataforma de despliegue:

- `WEATHER_API_KEY`: Clave de la API del clima. Puedes obtenerla registrándote de forma gratuita en [https://www.weatherapi.com](https://www.weatherapi.com).
- `MISTRAL_API_KEY`: Clave de la API de Mistral AI. Puedes generarla desde tu panel de control en [https://admin.mistral.ai/](https://admin.mistral.ai/).
- 

### Uso en local
Si vas a ejecutar el proyecto en tu máquina local, crea un archivo llamado `.env` en la raíz del proyecto con el siguiente contenido:

```env
WEATHER_API_KEY=tu_clave_de_weatherapi_aqui
MISTRAL_API_KEY=tu_clave_de_mistral_aqui
AMP_OTEL_ENDPOINT=endpoint_otel
AMP_AGENT_API_KEY=clave_otel
```

## Ejecución local

Asegúrate de tener instaladas las dependencias (FastAPI, uvicorn, langchain, langgraph, langchain_mistralai, requests, python-dotenv).
Instala las dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Luego, inicia el servidor:

```bash
set -a && source .env && set +a
amp-instrument uvicorn main:app --host 0.0.0.0 --port 8000
```

set -a/set +a exporta automáticamente todo lo que hay en .env (incluyendo las claves de Weather/Mistral) a la sesión de shell sin que tengas que escribir el token a mano.

Salir de entorno virtual:

```bash
deactivate
```

## Uso de la API

El servidor expone un endpoint principal para interactuar con el agente:

**Endpoint:** `POST /chat`

**Cuerpo de la petición (JSON):**
```json
{
  "session_id": "12345",
  "message": "¿Qué clima hace en Aranda de Duero y qué música me recomiendas para este tiempo?",
  "model": "ministral-3b-latest"
}
```

El `session_id` se utiliza como identificador de hilo (`thread_id`) para mantener la memoria y el contexto de la conversación de cada chat.
