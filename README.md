# movie-recommendation-chatbot

**Examples:**

*   **Using `curl`:**

    ```bash
    curl -k https://movie-recommendation-chatbot.haaretz.co.il/health
    ```
    *Expected Output (example):*
    ```json
    {"status":"ok","llm_client":"initialized"}
    ```


    ```bash
    curl -k -X POST https://movie-recommendation-chatbot.haaretz.co.il/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Recommend a science fiction movie from the 90s", "user_id": "user_123"}' \
    --no-buffer
    ```
    *Expected Output:* A stream of text chunks forming the chatbot's response.
