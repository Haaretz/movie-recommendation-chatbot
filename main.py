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

from config.loader import load_config
from logger import logger
from src.llm_api_client import LLMClient
from src.redis_chat_history import RedisChatHistory


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def create_llm_client_and_model():
    """
    Initialize AppConfig and LLMClient.
    """

    # Load full app config (llm, qdrant, embedding, fields)
    app_config = load_config()

    # Load system prompt from prompts.yaml
    import yaml

    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    sys_instruct = prompts.get("system_instructions", "")

    # Initialize LLM wrapper
    llm_client = LLMClient(
        llm_config=app_config.llm,
        embedding_config=app_config.embedding,
        qdrant_config=app_config.qdrant,
        fields_config=app_config.fields,
        sys_instruct=sys_instruct,
        redis_store=RedisChatHistory(),
    )

    # Also return raw genai client (used for token counting)
    model_client = genai.Client(api_key=app_config.llm.GOOGLE_API_KEY)

    return llm_client, model_client


# --------------------------------------------------------------------------- #
# One global instance, created at startup
# --------------------------------------------------------------------------- #
llm_client_instance, genai_client = create_llm_client_and_model()

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

    logger.debug("Received streaming request: '%s' for user %s", user_message, user_id)
    full_response = ""

    try:
        async for chunk in llm_client_instance.streaming_message(user_message, user_id):
            yield chunk
            await asyncio.sleep(0)
            full_response += chunk
        logger.debug("Final response: '%s'", full_response)

    except Exception as e:
        logger.error("LLMClient error: %s. Reinitializing client and retrying once.", e)
        llm_client_instance, client = create_llm_client_and_model()

        full_response_retry = ""
        async for chunk in llm_client_instance.streaming_message(user_message, user_id):
            yield chunk
            full_response_retry += chunk
        logger.debug("Final response after retry: '%s'", full_response_retry)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/chat", response_class=StreamingResponse)
async def handle_chat_stream(chat_message: ChatMessage = Body(...)):
    """
    POST /chat
    Validate the request and stream LLM responses in plain text.
    """
    logger.info("Received chat message: %s", chat_message)
    user_message = chat_message.message
    user_id = chat_message.user_id

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # ----------------------------------------------------------------------- #
    # Use the model name from our client instead of hard‑coding it
    # ----------------------------------------------------------------------- #
    chat_config = llm_client_instance.chat_config

    # Token limit check (hard limit: 150 tokens)
    token_count = genai_client.models.count_tokens(
        model=llm_client_instance.model_name, contents=user_message
    ).total_tokens

    if token_count > chat_config.token_limit:
        return chat_config.long_request

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
    logger.info("version: %s", "0.0.1")
    uvicorn.run(app, host="0.0.0.0", port=port)
