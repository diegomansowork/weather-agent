import dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# CAMBIO: Como ambos archivos están en la misma carpeta, importamos directamente de 'agent'
from agent import MODELO_POR_DEFECTO, create_agent

# Cargar variables de entorno
dotenv.load_dotenv()

# Inicializar la aplicación
app = FastAPI(title="Weather & Music AI Agent")

# Cache de agentes por modelo: evita reconstruir el grafo/LLM en cada request.
# La memoria (checkpointer) es compartida entre todos, así que cambiar de
# modelo a mitad de conversación no pierde el historial del thread_id.
_agentes_por_modelo: dict[str, object] = {}

def get_agent(model: str):
    if model not in _agentes_por_modelo:
        _agentes_por_modelo[model] = create_agent(model=model)
    return _agentes_por_modelo[model]

# Definir la estructura del payload esperado
class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = MODELO_POR_DEFECTO  # opcional: modelo de Mistral a utilizar

def run_agent(thread_id: str, question: str, model: str):
    agent_graph = get_agent(model)
    config = {
        "configurable": {
            # Los checkpoints (memoria) se acceden mediante este thread_id
            "thread_id": thread_id,
        }
    }

    # Transmitimos los eventos del grafo
    events = agent_graph.stream(
        {"messages": [("user", question)]}, config, stream_mode="values"
    )

    final_answer = None
    for event in events:
        if "messages" in event:
            final_answer = event["messages"][-1].content

    return final_answer

@app.post("/chat")
async def chat(payload: ChatRequest):
    result = {
        "response": run_agent(
            thread_id=payload.session_id,
            question=payload.message,
            model=payload.model,
        )
    }
    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)