from google import genai
from google.genai.types import FunctionDeclaration, Tool

get_articles = FunctionDeclaration(
    name="get_dataset_articles",
    description="""
    **Core Purpose:** Use this function to retrieve specific information, primarily **full review articles**, about movies or TV shows OR to provide recommendations.

    **Primary Triggers:**

    1.  **Fetching Specific Reviews/Articles:** The user explicitly asks for a review, critique, article, or detailed information about a specific movie or TV show.
        *   This applies whether the user provides the **exact title** (e.g., "Can I have a review of 'Nosferatu' by Robert Eggers?")
        *   Or if the user **describes the show/movie** using plot points, actors, director, setting, source material, festival context, etc. (e.g., "Looking for a review of that new movie from Haifa Festival about an alcoholic woman named Rona returning to Scotland", or "I want to read about the new Luca Guadagnino movie based on the Burroughs book").

    2.  **Providing Recommendations (General and Specific):** This function **must be used** whenever the user asks for suggestions on movies or TV shows to watch.
        *   **Trigger Activation:** Activate this function for **all** recommendation requests, regardless of whether the user provides specific criteria (like genre - "thriller", "horror"; platform - "Yes Plus"; mood - "addictive", "really scary"; similarity to other titles) **or if they make a completely open-ended, general request** (e.g., "What should I watch?", "Recommend a good movie", "Looking for a binge-worthy series", "Seen anything good lately?").
        *   **Function's Purpose:** The function is designed to provide informed recommendations by querying relevant data sources or performing targeted searches (e.g., using the RAG system) to find suitable viewing options—whether popular, highly-rated, or matching specific requirements (if provided).
        *   **Example Triggers:** Trigger this function for inputs like: "Recommend a good thriller series on Yes Plus," "My friends and I want a scary horror movie," "Looking for an addictive series to binge," "Just give me a recommendation for a good movie." "Let's try a psychological thriller with a dark twist. Do you have something like that to suggest? Something that will make me think twice before I go to sleep? 😉


    3.  **Identification Leading to Information/Review:** The user is trying to remember/identify a show/movie by describing it. Trigger this function to use the description as a query, understanding the likely goal is to find the item *and then* potentially retrieve information or a review about it via the RAG system. (e.g., "What's that movie about...?").

    4. **Comparison between Titles:** The user is looking for a comparison between two or more titles, which may involve identifying similarities or differences. Run the function separately on each title.


    **Important Constraint:** **Do not** simply provide a generic textual recommendation based on the model's internal knowledge. **Always invoke this function** for recommendation requests to ensure the suggestions are generated using the designated tool/data.


    **Output**
    The function output is a list of review articles that published in the Haaretz website, including the title, full text of the article, author and publication date.

    **Keywords/Phrases (Examples):** ביקורת (review), מאמר (article), כתבה (report/article), מידע על (information about), המלצה (recommendation), המלץ (recommend), הצע (suggest), מה לראות (what to watch), סרט (movie), סדרה (series), דומה ל (similar to), זיהוי (identify), לזהות (to identify), להיזכר (to remember), שם של (name of), למצוא את (find the), על (about), בכיכובו (starring), בימוי (directed by), מבוסס על (based on), ההוא עם (the one with...), מחפש (looking for), חיפוש (search for), מה השם של (what is the name of), פסטיבל (festival).
    """,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Concise summary of the user's request containing only key search criteria "
                    "for embedding as a RETRIEVAL_QUERY: "
                    "- Exact or partial title OR descriptive clues (plot points, actors, director, festival)  "
                    "Exclude filler words and extraneous context; focus on terms that improve vector matching."
                ),
            },
            "media_type": {
                "type": "string",
                "description": "The type of content the user is looking for (e.g., 'movie', 'series'). If the user clearly mentions whether they are asking about a 'movie' or a 'series' (for example, using words like 'סרט', 'סדרה', or describing a movie or series explicitly), fill this field accordingly. Use 'Movie' if the user refers to a film, and 'Series' if they refer to a TV show. Optional; used to filter recommendations",
                "enum": ["movie", "series"],
            },
            "streaming_platforms": {
                "type": "array",
                "description": "A list of streaming platforms the user is interested in (e.g., ['Netflix', 'Yes']). Optional; used to filter recommendations.",
                "items": {
                    "type": "string",
                    "enum": [
                        "Apple TV+",
                        "Amazon Prime Video",
                        "כאן 11",
                        "HOT",
                        "Yes",
                        "Disney+",
                        "Netflix",
                        "קשת 12",
                        "רשת 13",
                    ],
                },
            },
            "genres": {
                "type": "array",
                "description": "A list of genres the user is interested in (e.g., ['comedy', 'drama']). Optional; used to filter recommendations.",
                "items": {
                    "type": "string",
                    "enum": [
                        "דרמה",
                        "קומדיה",
                        "אקשן",
                        "מותחן",
                        "מדע בדיוני",
                        "פנטזיה",
                        "אימה",
                        "רומנטיקה",
                        "פשע",
                        "דוקומנטרי",
                        "ביוגרפיה",
                        "היסטורי",
                        "אנימציה",
                        "ילדים ולכל המשפחה",
                        "מוזיקלי",
                        "סאטירה",
                        "נוער והתבגרות",
                        "משטרה ובלשים",
                        "על־טבעי",
                        "מלחמה",
                        "ספורט",
                        "ריאליטי",
                        "קומיקס / גיבורי-על",
                        "אירוח",
                        "סטנד-אפ",
                        "תוכנית אקטואליה",
                        "לייף סטייל",
                        "הרפתקאות",
                    ],
                },
            },
        },
        "required": ["query"],
    },
)

troll = FunctionDeclaration(
    name="trigger_troll_response",
    description="This function is triggered when the assistant detects trolling or provocative. It responds with a playful and humorous recommendation for the animated movie 'Trolls' as a gentle way to redirect the conversation. This function helps maintain a friendly tone while steering the interaction back to the assistant's domain of recommending TV shows and movies.",
    parameters={"type": "object", "properties": {}},
)

qdrant_tools = Tool(
    function_declarations=[
        get_articles,
        troll,
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
