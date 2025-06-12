# movie-recommendation-chatbot

A **streaming recommendation assistant** for movies & TV built with **FastAPI, Google Gemini, Qdrant vector search, and Redis chat history**.

> **Note:** Requests to `/chat` and `/regenerate` endpoints require a valid `sso_token` cookie (Base64‑encoded JSON with `userType` set to `paying`). Non‑paying or missing tokens will receive a friendly upgrade message.

> **Pre-processing Note:** The document and embedding pre-processing pipeline for this project is available in the companion repository: [`ask-haaretz-rag-pipeline`](https://github.com/haaretz/ask-haaretz-rag-pipeline) (private/internal).

---

## 🔧 Endpoints & Examples

### ✅ Health check

```bash
curl -k https://movie-recommendation-chatbot.haaretz.co.il/health
```

*Expected Output:*

```json
{"status":"ok","llm_client":"initialized"}
```

---

### 💬 New chat

```bash
export TOKEN=$(echo -n '{"userId": "user_123", "userType": "paying"}' | base64)

curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -H "Cookie: sso_token=$TOKEN" \
  -d '{"message": "מה שלומך?", "session_id": "session_abc"}'
```

*Expected Output:*
A stream of text chunks forming the chatbot's response.

---

### ♻️ Regenerate response

```bash
curl -X POST https://movie-recommendation-chatbot.haaretz.co.il/regenerate \
  -H "Content-Type: application/json" \
  -H "Cookie: sso_token=$TOKEN" \
  -d '{"session_id": "session_abc"}'
```

*Expected Output:*
A stream of text chunks forming the chatbot's regenerated response, based on the last chat history.
