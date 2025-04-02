import uvicorn
import os
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

from src.llm_api_client import LLMClient 
from config.load_config import load_config 
from logger import logger 

prompts = load_config("config/prompts.yaml")
config = load_config("config/config.yaml")
logger.info("Configuration files loaded successfully.")

sys_instruct = prompts.get("system_instructions")
api_key = config.get("llm", {}).get("GOOGLE_API_KEY")
model_name = config.get("llm", {}).get("llm_model_name")

llm_client_instance = LLMClient(
    model_name=model_name,
    api_key=api_key,
    sys_instruct=sys_instruct,
    config=config 
)

class ChatMessage(BaseModel):
    message: str

app = FastAPI(
    title="LLM Streaming Chat API",
    description="API for interacting with the LLM via streaming.",
    version="1.0.0"
)

async def stream_llm_response(user_message: str) -> AsyncGenerator[str, None]:
    """
    Asynchronous generator that yields chunks from the LLM's streaming response.
    Handles function calls internally via the LLMClient.
    """
    logger.info(f"Received streaming request: '{user_message}'")
    async for chunk in llm_client_instance.streaming_message(user_message):
        if isinstance(chunk, str):
            yield chunk


@app.post("/chat", response_class=StreamingResponse)
async def handle_chat_stream(chat_message: ChatMessage = Body(...)):

    user_message = chat_message.message
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return StreamingResponse(
        stream_llm_response(user_message),
        media_type="text/plain" 
    )

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Returns 'ok' if the service is running and the LLM client was initialized.
    """
    if llm_client_instance:
        return {"status": "ok", "llm_client": "initialized"}
    else:
        raise HTTPException(status_code=503, detail={"status": "error", "llm_client": "not_initialized"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080)) 
    logger.info(f"Starting Uvicorn server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)