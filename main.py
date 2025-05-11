"""
Full FastAPI server file – revised so the model name used for
`count_tokens()` comes from configuration instead of being hard‑coded.
"""

import asyncio
import os
from typing import AsyncGenerator

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from pydantic import BaseModel

from config.load_config import load_config
from logger import logger
from src.llm_api_client import LLMClient
from src.redis_chat_history import RedisChatHistory


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def create_llm_client():
    """
    (Re)initialize the LLMClient with current config and return it
    together with its genai client.
    """
    prompts = load_config("config/prompts.yaml")
    sys_instruct = prompts.get("system_instructions")

    config = load_config("config/config.yaml")
    llm_cfg = config.get("llm", {})

    redis_store = RedisChatHistory()

    # Grab the model name and API key from config
    model_name = llm_cfg.get("llm_model_name")
    api_key = llm_cfg.get("GOOGLE_API_KEY")

    # Initialise Google genai client
    client = genai.Client()

    # Construct our wrapper client
    llm_client = LLMClient(
        model_name=model_name,
        api_key=api_key,
        sys_instruct=sys_instruct,
        config=config,
        redis_store=redis_store,
    )
    return llm_client, client


# --------------------------------------------------------------------------- #
# One global instance, created at startup
# --------------------------------------------------------------------------- #
llm_client_instance, client = create_llm_client()

# --------------------------------------------------------------------------- #
# FastAPI setup
# --------------------------------------------------------------------------- #
origins = [
    "https://localhost",
    "https://localhost.haaretz.co.il",
    "https://localhost:3000",
    "https://react-stage.haaretz.co.il",
    "https://canary.haaretz.co.il",
    "https://react-cu-86c2jh5hv.k8s-stage.haaretz.co.il",
]


class ChatMessage(BaseModel):
    """Schema for a single chat message arriving from the front‑end."""

    message: str
    user_id: str


app = FastAPI(
    title="LLM Streaming Chat API",
    description="API for interacting with the LLM via streaming.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Streaming helper
# --------------------------------------------------------------------------- #
async def stream_llm_response(user_message: str, user_id: str) -> AsyncGenerator[str, None]:
    """
    Yield chunks of the LLM streaming response.
    If an error occurs, rebuild the client once and retry.
    """
    global llm_client_instance, client

    logger.info("Received streaming request: '%s' for user %s", user_message, user_id)
    full_response = ""

    try:
        async for chunk in llm_client_instance.streaming_message(user_message, user_id):
            yield chunk
            await asyncio.sleep(0)
            full_response += chunk
        logger.info("Final response: '%s'", full_response)

    except Exception as e:
        logger.error("LLMClient error: %s. Reinitializing client and retrying once.", e)
        llm_client_instance, client = create_llm_client()

        full_response_retry = ""
        async for chunk in llm_client_instance.streaming_message(user_message, user_id):
            yield chunk
            full_response_retry += chunk
        logger.info("Final response after retry: '%s'", full_response_retry)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/chat", response_class=StreamingResponse)
async def handle_chat_stream(chat_message: ChatMessage = Body(...)):
    """
    POST /chat
    Validate the request and stream LLM responses in plain text.
    """
    user_message = chat_message.message
    user_id = chat_message.user_id

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # ----------------------------------------------------------------------- #
    # Use the model name from our client instead of hard‑coding it
    # ----------------------------------------------------------------------- #
    model_for_count = llm_client_instance.model_name

    # Token limit check (hard limit: 150 tokens)
    token_count = client.models.count_tokens(model=model_for_count, contents=user_message).total_tokens

    if token_count > 150:
        raise HTTPException(
            status_code=413,
            detail=f"Message too long ({token_count} tokens). Maximum is 150.",
        )

    return StreamingResponse(
        stream_llm_response(user_message, user_id),
        media_type="text/plain",
    )


@app.get("/health")
async def health_check():
    """
    GET /health
    Basic health check to verify the service and client are up.
    """
    if llm_client_instance:
        return {"status": "ok", "llm_client": "initialized"}
    raise HTTPException(status_code=503, detail={"status": "error", "llm_client": "not_initialized"})


@app.get("/version")
async def version():
    """
    GET /version
    Returns the API version.
    """
    return {"version": "0.0.1"}


# --------------------------------------------------------------------------- #
# Entry‑point when running as a script
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting Uvicorn server on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
