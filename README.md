# movie-recommendation-chatbot
A **streaming recommendation assistant** for movies & TV built with **FastAPI, Google Gemini, Qdrant vector search and Redis chat history**.

**Examples:**

*   **Using `curl`:**

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
    -d '{"message": "Recommend a science fiction movie from the 90s", "user_id": "user_123"}' \
    --no-buffer
    ```
    *Expected Output:* A stream of text chunks forming the chatbot's response.

    ### Regenerate response
    ```bash
    curl -k -X POST https://movie-chat.stage.haaretz.co.il/regenerate \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user_123"}' \
     --no-buffer
     ```
     *Expected Output* A stream of text chunks forming the chatbot's response, following the last message in the chat history.
