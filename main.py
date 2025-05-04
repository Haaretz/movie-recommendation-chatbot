import os
from typing import AsyncGenerator

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.load_config import load_config
from logger import logger
from src.llm_api_client import LLMClient
from src.redis_chat_history import RedisChatHistory


def create_llm_client() -> LLMClient:
    """
    (Re)initialize the LLMClient with current config.
    """
    prompts = load_config("config/prompts.yaml")
    sys_instruct = prompts.get("system_instructions")
    config = load_config("config/config.yaml")
    llm_cfg = config.get("llm", {})

    redis_store = RedisChatHistory()
    return LLMClient(
        model_name=llm_cfg.get("llm_model_name"),
        api_key=llm_cfg.get("GOOGLE_API_KEY"),
        sys_instruct=sys_instruct,
        config=config,
        redis_store=redis_store,
    )


# Initialize the client once at startup
llm_client_instance = create_llm_client()

origins = [
    "https://localhost",
    "https://localhost.haaretz.co.il",
    "https://localhost:3000",
    "https://react-stage.haaretz.co.il",
    "https://canary.haaretz.co.il",
]


class ChatMessage(BaseModel):
    message: str
    user_id: str


app = FastAPI(
    title="LLM Streaming Chat API", description="API for interacting with the LLM via streaming.", version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


async def stream_llm_response(user_message: str, user_id: str) -> AsyncGenerator[str, None]:
    """
    Asynchronous generator that yields chunks from the LLM's streaming response.
    If an error occurs, it will re-create the LLMClient and retry once.
    """
    global llm_client_instance

    # First attempt
    logger.info(f"Received streaming request: '{user_message}' for user {user_id}")
    txt = ""
    try:
        async for chunk in llm_client_instance.streaming_message(user_message, user_id):
            yield chunk
            txt += chunk
        logger.info(f"Final response: '{txt}'")

    except Exception as e:
        # On failure, log, rebuild client, and retry one more time
        logger.error(f"LLMClient error: {e}. Reinitializing client and retrying once.")
        llm_client_instance = create_llm_client()

        txt_retry = ""
        async for chunk in llm_client_instance.streaming_message(user_message, user_id):
            yield chunk
            txt_retry += chunk
        logger.info(f"Final response after retry: '{txt_retry}'")


@app.post("/chat", response_class=StreamingResponse)
async def handle_chat_stream(chat_message: ChatMessage = Body(...)):
    """
    POST /chat
    Streams back LLM responses as plain text.
    """
    user_message = chat_message.message
    user_id = chat_message.user_id

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    return StreamingResponse(stream_llm_response(user_message, user_id), media_type="text/plain")


@app.get("/health")
async def health_check():
    """
    GET /health
    Simple health check to verify the service and client are up.
    """
    if llm_client_instance:
        return {"status": "ok", "llm_client": "initialized"}
    else:
        raise HTTPException(status_code=503, detail={"status": "error", "llm_client": "not_initialized"})


@app.get("/version")
async def version():
    """
    GET /version
    Returns the version of the API.
    """
    return {"version": "0.0.1"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Uvicorn server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
