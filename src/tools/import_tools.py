from google import genai
from google.genai.types import FunctionDeclaration, Tool

from config.loader import load_config

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

    4.  **Writer-Specific Filtering:** If the user explicitly asks to read reviews or get recommendations by a particular writer — for example, "What does חן חדד recommend?" or "Show me movies reviewed by ניב הדס" — include the relevant writer's name in the `writer_filter` parameter.
        *   Only apply this filter if the user refers clearly to one or more writers by name.
        *   This filter can also be used in combination with genre or platform filters to find personalized recommendations by trusted voices.


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
            "writer_filter": {
                "type": "array",
                "description": "Optional filter for specific article authors. Use this when the user explicitly asks for reviews or recommendations by one or more named writers (e.g., חן חדד).",
                "items": {
                    "type": "string",
                    "enum": [
                        "חן חדד",
                        "אורון שמיר",
                        "אורי קליין",
                        "נתנאל שלומוביץ",
                        "שני ליטמן",
                        "פבלו אוטין",
                        "ניב הדס",
                        "גילי איזיקוביץ",
                    ],
                },
            },
            # "best_of": {
            #     "type": "boolean",
            #     "description": (
            #         "Optional flag to indicate if the user is looking for 'best of' recommendations. "
            #         "Set to true if the user asks for top-rated or highly recommended content, "
            #         "like 'best movies' or 'top series to watch'."
            #         "This helps in filtering for high-quality recommendations. (5 stars recommendations)"
            #     ),
            # },
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
    # Load configuration
    app_config = load_config()

    # Use system instructions from prompts YAML (optional)
    try:
        import yaml

        with open("config/prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)
            sys_instruct = prompts.get("system_instructions", "")
    except Exception as e:
        sys_instruct = ""
        print(f"Could not load system instructions: {e}")

    # Initialize Gemini client using API key from config
    client = genai.Client(api_key=app_config.llm.GOOGLE_API_KEY)

    # Use model name from config
    MODEL_ID = app_config.llm.llm_model_name

    # Create chat with tools
    chat = client.chats.create(
        model=MODEL_ID,
        config=genai.types.GenerateContentConfig(
            system_instruction=sys_instruct,
            tools=[qdrant_tools],
        ),
    )

    # Example user prompt
    prompt = """
שלום, קוראים לי משה, בא לי לראות סרט מצויר בסגנון פיקסר. יש משהו שקליין המליץ עליו?
    """
    response = chat.send_message(prompt)

    # Print the function call(s) chosen by the LLM
    print(response.function_calls)
    print(response.function_calls[0].args["writer_filter"])
