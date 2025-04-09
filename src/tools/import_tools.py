from google import genai
from google.genai.types import Tool

get_articles = {
    "name": "recommendations_for_tv_and_movies",
    "description": """
    Primary Trigger: Use this function when the user wants movie or TV show recommendations, OR when they are trying to identify/recall a specific movie/TV show based on details they provide.

    This includes:
    1. Explicit requests for suggestions (e.g., "Recommend a sci-fi series", "What should I watch?", "Suggest movies like Dune").
    2. Attempts to find a specific title by describing its plot, actors, director, setting, source material, or other identifying features (e.g., "What's that movie about a robot falling in love?", "I'm trying to remember the name of a series with Bryan Cranston before Breaking Bad", "The movie directed by Villeneuve with giant worms").
    3. Requests for information (like reviews, details) about a specific movie/show, even if the user doesn't know the exact title but provides descriptive clues (e.g., "I'm looking for a review of the new Luca Guadagnino movie, the one based on a Burroughs book about a gay man in Mexico in the 50s").

    Keywords/Phrases (Examples): recommend, suggest, watch, movie, series, TV show, like, similar to, review, identify, remember, name of, find the movie/show, about, starring, directed by, based on, the one with..., looking for, search for, what is the name of.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's request in Hebrew, containing either the criteria for recommendations OR the descriptive clues for identifying a specific movie/show. This should capture the essence of what they are looking for or trying to identify.",
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
שלום, קוראים לי משה, בא לי לראות סרט מצויר בסגנון פיקסר
    """
    response = chat.send_message(prompt)
    response.function_calls
    print(response.function_calls)
