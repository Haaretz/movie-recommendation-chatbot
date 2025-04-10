from google import genai
from google.genai.types import Tool

get_articles = {
    "name": "find_movie_tv_info_or_review",
    "description": """
    **Core Purpose:** Use this function to retrieve specific information, primarily **full review articles**, about movies or TV shows using a Retrieval-Augmented Generation (RAG) system, OR to provide recommendations.

    **Primary Triggers:**

    1.  **Fetching Specific Reviews/Articles:** The user explicitly asks for a review, critique, article, or detailed information about a specific movie or TV show.
        *   This applies whether the user provides the **exact title** (e.g., "Can I have a review of 'Nosferatu' by Robert Eggers?")
        *   Or if the user **describes the show/movie** using plot points, actors, director, setting, source material, festival context, etc. (e.g., "Looking for a review of that new movie from Haifa Festival about an alcoholic woman named Rona returning to Scotland", or "I want to read about the new Luca Guadagnino movie based on the Burroughs book").
        *   The function should be triggered to pass the user's query to the RAG system to find the relevant article.

    2.  **Recommendations:** The user asks for suggestions on what to watch, potentially based on preferences, genres, or similarity to other titles (e.g., "Recommend a sci-fi series").

    3.  **Identification Leading to Information/Review:** The user is trying to remember/identify a show/movie by describing it. Trigger this function to use the description as a query, understanding the likely goal is to find the item *and then* potentially retrieve information or a review about it via the RAG system. (e.g., "What's that movie about...?").

    **Keywords/Phrases (Examples):** ביקורת (review), מאמר (article), כתבה (report/article), מידע על (information about), המלצה (recommendation), המלץ (recommend), הצע (suggest), מה לראות (what to watch), סרט (movie), סדרה (series), דומה ל (similar to), זיהוי (identify), לזהות (to identify), להיזכר (to remember), שם של (name of), למצוא את (find the), על (about), בכיכובו (starring), בימוי (directed by), מבוסס על (based on), ההוא עם (the one with...), מחפש (looking for), חיפוש (search for), מה השם של (what is the name of), פסטיבל (festival).
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
