from src.llm_api_client import LLMClient
import fastapi
from config.load_config import load_config



app = fastapi.FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "0.0.1"}

config = load_config("config/config.yaml")
prompts = load_config("config/prompts.yaml")
model_name = config["llm"]["llm_model_name"]
api_key = config["llm"].get("GOOGLE_API_KEY")
sys_instruct = prompts["system_instructions"]
client = LLMClient(model_name=model_name, api_key=api_key, sys_instruct=sys_instruct, config=config)

@app.post("/chat")
def chat(message: str):
    response = client.send_message(message).text
    return response

@app.post("/reset")
def reset():
    client.reset_chat_session()
    return {"status": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
