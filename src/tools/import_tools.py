import pandas as pd
from google import genai
from google.genai.types import Tool


get_articles = {
    "name": "recommendations_for_tv_and_movies",
    "description": """
    Use this function when the user is asking for recommendations for TV and movie shows.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "query to retrieve relevant articles",
            },
        },
        "required": ["query"],
    },
}



qdrant_tools = Tool(
    function_declarations=[
        get_articles,
    ],
)

if __name__ == "__main__":
    import os

    from google.genai import types

    from config.load_config import load_config

    prompts = load_config("config/prompts.yaml")
    sys_instruct = prompts["system_instructions"]
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    MODEL_ID = "gemini-2.0-flash"

    chat = client.chats.create(
        model=MODEL_ID,
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct,
            tools=[qdrant_tools],
        ),
    )

    prompt = """
    סרטי ילדים מצויירים טובים
    """
    response = chat.send_message(prompt)
    response.function_calls
    print(response.function_calls)
