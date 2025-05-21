# movie-recommendation-chatbot

A **streaming recommendation assistant** for movies & TV built with **FastAPI, Google Gemini, Qdrant vector search and Redis chat history**.

> **Note:** Requests to `/chat` and `/regenerate` endpoints require a valid `sso_token` cookie (Base64‑encoded JSON with `userType` set to `paying`). Non‑paying or missing tokens will receive a friendly upgrade message.

**Examples:**

* **Using `curl`:**

  ### health check

  ```bash
  curl -k https://movie-recommendation-chatbot.haaretz.co.il/health
  ```

  *Expected Output (example):*

  ```json
  {"status":"ok","llm_client":"initialized"}
  ```

  ### New chat

  ```bash
  curl -k -X POST https://movie-recommendation-chatbot.haaretz.co.il/chat \
    -H "Content-Type: application/json" \
    -H "Cookie: sso_token=<BASE64_ENCODED_TOKEN>" \
    -d '{"message": "Recommend a science fiction movie from the 90s", "session_id": "user_123"}' \
    --no-buffer
  ```

  *Expected Output:* A stream of text chunks forming the chatbot's response.

  ### Regenerate response

  ```bash
  curl -k -X POST https://movie-recommendation-chatbot.haaretz.co.il/regenerate \
    -H "Content-Type: application/json" \
    -H "Cookie: sso_token=<BASE64_ENCODED_TOKEN>" \
    -d '{"session_id": "user_123"}' \
    --no-buffer
  ```

  *Expected Output:* A stream of text chunks forming the chatbot's regenerated response, based on the last chat history.
