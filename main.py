"""
Full FastAPI server file – revised so the model name used for
`count_tokens()` comes from configuration instead of being hard-coded,
and to enforce that only paying users may call /chat and /regenerate.
"""

import asyncio
import base64
import json
import os
from http.cookies import SimpleCookie
from typing import AsyncGenerator

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request
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
        redis_store=RedisChatHistory(app_config.chat.chat_ttl_seconds),
        chat_config=app_config.chat,
    )

    # Also return raw genai client (used for token counting)
    model_client = genai.Client(api_key=app_config.llm.GOOGLE_API_KEY)

    return llm_client, model_client


# --------------------------------------------------------------------------- #
# Enforce “paying” user (helper)
# --------------------------------------------------------------------------- #
def get_sso_token_data(request: Request) -> dict | None:
    """
    Extracts and decodes the sso_token if present.
    Returns the decoded token data or None if no token is present.
    Raises 400 only if the token exists but is badly formatted.
    """
    cookie_header = request.headers.get("cookie")
    if not cookie_header:
        return None

    cookie = SimpleCookie()
    cookie.load(cookie_header)
    if "sso_token" not in cookie:
        return None

    token_value = cookie["sso_token"].value
    try:
        decoded_str = base64.b64decode(token_value).decode("utf-8")
        token_data = json.loads(decoded_str)
        return token_data
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Invalid sso_token format")


# --------------------------------------------------------------------------- #
# One global instance, created at startup
# --------------------------------------------------------------------------- #
llm_client_instance, genai_client = create_llm_client_and_model()

# --------------------------------------------------------------------------- #
# FastAPI setup
# --------------------------------------------------------------------------- #
origins = [
    "https://localhost",
    "https://local.haaretz.co.il",
    "https://localhost.haaretz.co.il",
    "https://localhost:3000",
    "https://react-stage.haaretz.co.il",
    "https://canary.haaretz.co.il",
    "https://react-cu-86c2jh5hv.k8s-stage.haaretz.co.il",
]


class ChatMessage(BaseModel):
    """Schema for a single chat message arriving from the front-end."""

    message: str
    session_id: str


class UserIdOnly(BaseModel):
    """Schema for regenerate endpoint."""

    session_id: str


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
async def stream_llm_response(user_message: str, sso_id: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Yield chunks of the LLM streaming response.
    If an error occurs, wait one second and retry indefinitely.
    """
    global llm_client_instance, genai_client

    while True:
        logger.debug("Streaming request: '%s' for user %s and session %s", user_message, sso_id, session_id)
        full_response = ""

        try:
            async for chunk in llm_client_instance.streaming_message(user_message, session_id):
                yield chunk
                await asyncio.sleep(0)
                full_response += chunk

            logger.debug("Final response: '%s'", full_response)
            return  # exit after success

        except Exception as e:
            logger.error("LLMClient error: %s. Reinitializing client and retrying.", e)
            llm_client_instance, genai_client = create_llm_client_and_model()
            await asyncio.sleep(1)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/chat", response_class=StreamingResponse, summary="Stream chat reply (paying users only)")
async def handle_chat_stream(
    request: Request,
    chat_message: ChatMessage = Body(...),
):
    """
    POST /chat
    Validate the request, enforce paying-user, then stream LLM responses in plain text.
    """
    token_data = get_sso_token_data(request)
    logger.debug("SSO token payload:", extra={"token_data": token_data})
    sso_id = token_data["userId"] if token_data else None
    chat_config = llm_client_instance.chat_config

    # If not a paying user, return friendly upgrade message
    if not token_data or token_data.get("userType") != "paying":
        return StreamingResponse(
            iter([chat_config.non_paying_messages]),
            media_type="text/plain",
            status_code=200,
        )

    user_message = chat_message.message
    session_id = chat_message.session_id

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Token-limit check using configured model_name
    token_count = genai_client.models.count_tokens(
        model=llm_client_instance.model_name, contents=user_message
    ).total_tokens

    if token_count > chat_config.token_limit:
        return chat_config.long_request

    return StreamingResponse(
        stream_llm_response(user_message, sso_id, session_id),
        media_type="text/plain",
    )


@app.post("/regenerate", response_class=StreamingResponse, summary="Regenerate last reply (paying users only)")
async def handle_regenerate(
    request: Request,
    user_data: UserIdOnly = Body(...),
):
    """
    POST /regenerate
    Regenerates the last assistant message based on the last user input.
    """
    token_data = get_sso_token_data(request)
    chat_config = llm_client_instance.chat_config

    # If not a paying user, return friendly upgrade message
    if not token_data or token_data.get("userType") != "paying":
        return StreamingResponse(
            iter([chat_config.non_paying_messages]),
            media_type="text/plain",
            status_code=200,
        )

    session_id = user_data.session_id
    sso_id = token_data["userId"]

    return StreamingResponse(
        llm_client_instance.regenerate_response(sso_id, session_id),
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
    return {"version": "0.1.0"}


# --------------------------------------------------------------------------- #
# Entry-point when running as a script
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting Uvicorn server on port %s", port)
    logger.info("version: %s", "0.0.1")
    uvicorn.run(app, host="0.0.0.0", port=port)
