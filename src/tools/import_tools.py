import pandas as pd
from google import genai
from google.genai.types import FunctionDeclaration, Tool

write_name = pd.read_csv(r"data/writer_names.csv")
write_names_list = write_name[write_name["unique_article_count"] > 30][
    "writer_name"
].tolist()  # TODO: need to change to denmic list
section_primary = pd.read_csv(r"data/section_primary.csv")["section_primary"].tolist()
section_secondary = pd.read_csv(r"data/section_secondary.csv")["section_secondary"].tolist()


get_articles = {
    "name": "get_articles",
    "description": """
    Use this function when the user is asking for information, opinions or recommendations.
    Extract from the user input the prompt to retrieve relevant articles.
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


# filter_writer_names = FunctionDeclaration(
#     name="filter_writer_names",
#     description="""
#     Filter by writer's name. Examples of cases where this function should be used:
#     * "According to **Merav Arlozorov**, what taxes are going to increase in 2023?"
#     * "In **Amos Harel's** article, what did the Minister of Defense answer him regarding the ceasefire question?"
#     * "What did **Dan Zaken** write about the nuclear deal with Iran?"
#     """,
#     parameters={
#         "type": "OBJECT",
#         "properties": {
#             "writer_names": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of writer names to filter by"},
#         },
#     },
# )


filter_brand = FunctionDeclaration(
    name="filter_brand",
    description="Filter by brand",
    parameters={
        "type": "OBJECT",
        "properties": {
            "brand": {
                "type": "STRING",
                "description": "Newspaper brand",
                "enum": ["הארץ", "דה מרקר"],
            },
        },
    },
)


filter_writer_names = FunctionDeclaration(
    name="named_entity_recognition_for_writer_names",
    description="""
    Named-entity recognition (NER) tool to filter articles by writer names.
    Find articles written by one or more specific writers.
    """,
    parameters={
        "type": "OBJECT",
        "properties": {
            "writer_names": {
                "type": "ARRAY",
                "description": "List of writer names to filter by",
                "items": {
                    "type": "STRING",
                    "enum": write_names_list,
                },
            },
        },
    },
)

# filter_time_range = FunctionDeclaration(
#     name="filter_articles_by_time_range",
#     description=f"""Filter articles by publication time range when two specific dates are given for filtering.
#                     Today date is {datetime.datetime.today().strftime('%Y-%m-%d')}. You need to find the time entity, and calculate the date in the format 'yyyy-mm-dd'.
#                     This tool is suitable for a wide range of queries, including:
#                     * "בכמה עלתה הריבית בשנת **2023**?" -> publish_time_start = 2023-01-01, publish_time_end = 2023-12-31
#                     * "תסכם את המשבר סביב הנפט באמריקה הלטינית שהתרחש **במרץ **". -> publish_time_start = {datetime.datetime(datetime.datetime.today().year - 1, 3, 1).strftime('%Y-%m-%d')} , publish_time_end = {datetime.datetime(datetime.datetime.today().year - 1, 3, 31).strftime('%Y-%m-%d')}
#                     * "מירב ארלוזרוב כתבה **ביולי אוגוסט** כתבה על הכלכלן הראשי, תוכל למצוא אותה?" -> publish_time_start = {datetime.datetime(datetime.datetime.today().year, 7, 1).strftime('%Y-%m-%d')}, publish_time_end = {datetime.datetime(datetime.datetime.today().year, 8, 31).strftime('%Y-%m-%d')}
#                     """,
#     parameters={
#         "type": "OBJECT",
#         "properties": {
#             "publish_time_start": {
#                 "type": "STRING",
#                 "description": """Start date for filtering a specific time range.
#                                  Data format:
#                                  - 'yyyy-mm-dd' (full date, for example: '2023-10-26').""",
#             },
#             "publish_time_end": {
#                 "type": "STRING",
#                 "description": """End date for filtering a specific time range.
#                                  Data format:
#                                  - 'yyyy-mm-dd' (full date, for example: '2024-10-26').""",
#             },
#         },
#         "required": ["publish_time_start", "publish_time_end"],
#     },
# )


# filter_time_last_day = FunctionDeclaration(
#     name="filter_articles_last_day",
#     description="""Filter articles published in the last day.
#     This tool is suitable for a wide range of queries, including:
#     * "חפש מאמרים שפורסמו היום על פוליטיקה."
#     * "מה חדש היום בתחום הטכנולוגיה?"
#     * "הצג לי רק מאמרים שפורסמו היום בנושא בריאות."
#     * "מה כתבו העיתונאים היום על המשבר הכלכלי?"
#     Use this tool when the user is requesting very recent or same-day information.
#     """,
#     parameters={
#         "type": "OBJECT",
#         "properties": {
#             "last_day": {
#                 "type": "BOOLEAN",
#                 "description": "Filter articles published in the last day",
#             },
#         },
#         "required": ["last_day"],
#     },
# )


# filter_time_last_week = FunctionDeclaration(
#     name="filter_articles_last_week",
#     description="""
#     Filter articles published in the last week.
#     This tool is suitable for a wide range of queries, including:
#     * "חפש מאמרים שפורסמו בשבוע האחרון על טכנולוגיה."
#     * "מה היו הכותרות הראשיות בשבוע האחרון בנושא כלכלה?"
#     * "הצג לי רק מאמרים חדשים מהשבוע האחרון על ספורט."
#     * "על איזה סרטים המליצה גילי איזיקוביץ השבוע?"
#     Use this tool when the user is requesting up-to-date or limited recent information.
#     """,
#     parameters={
#         "type": "OBJECT",
#         "properties": {
#             "last_week": {
#                 "type": "BOOLEAN",
#                 "description": "Should articles from the last week be filtered?",
#             },
#         },
#         "required": ["last_week"],
#     },
# )


# filter_time_last_month = FunctionDeclaration(
#     name="filter_articles_last_month",
#     description="""Filter articles published in the last month.
#     This tool is suitable for a wide range of queries, including:
#     * "חפש מאמרים שפורסמו בחודש האחרון על בינה מלאכותית."
#     * "מה היו ההתפתחויות האחרונות בחודש האחרון בתחום הסייבר?"
#     * "הצג לי רק מאמרים חדשים מהחודש האחרון בנושא חלל."
#     * "על איזה סרטים המליצה גילי איזיקוביץ החודש?"
#     Use this tool when the user is requesting up-to-date or limited recent information.
#     """,
#     parameters={
#         "type": "OBJECT",
#         "properties": {
#             "last_month": {
#                 "type": "BOOLEAN",
#                 "description": "Should articles from the last month be filtered?",
#             },
#         },
#         "required": ["last_month"],
#     },
# )


filter_primary_section = FunctionDeclaration(
    name="filter_primary_section",
    description="Filter by category. Only enable if you are explicitly told to filter by category.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "primary_section": {
                "type": "ARRAY",
                "items": {"type": "STRING", "enum": section_primary},
                "description": "List of categories",
            },
        },
    },
)


filter_secondary_section = FunctionDeclaration(
    name="filter_secondary_section",
    description="Filter by subcategory. Only enable if you are explicitly one of the subcategory is mentioned.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "secondary_section": {
                "type": "ARRAY",
                "items": {"type": "STRING", "enum": section_secondary},
                "description": "List of secondary categories",
            },
        },
    },
)

# filter_tags = FunctionDeclaration(
#     name="filter_tags",
#     description="Filter by tags. Only enable if explicitly told to filter by tags",
#     parameters={
#         "type": "OBJECT",
#         "properties": {
#             "tags": {
#                 "type": "ARRAY",
#                 "items": {"type": "STRING"},
#                 "description": "List of tags to filter",
#             },
#         },
#     },
# )

# filter_article_type = {
#     "name": "filter_article_type",
#     "description": """
#     Filters articles based on their type, such as reviews, recipes, or opinion pieces.
#     This function is useful when the user is asking for a specific type of article.

#     Examples:
#     * "Show me movie reviews."
#     * "Find recipes for chocolate cake."
#     * "I want to read opinion articles about politics."
#     * "Recommendations for new restaurants." (Implies review/recommendation type)
#     * "Recipes for pasta." (Implies recipe type)
#     * "Opinion pieces on climate change." (Implies opinion type)
#     """,
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "article_type": {
#                 "type": "string",
#                 "description": "The type of article to filter by.",
#                 "enum": list(ARTICLE_TYPE_TRANSLATION.keys()),
#             },
#         },
#     },
# }

# url = FunctionDeclaration(
#     name="analyze_article_content",
#     description="""
#     **חילוץ וניתוח תוכן ממאמר בודד הנתון בקישור URL.**
#     פונקציה זו מיועדת **לעיבוד מאמר ספציפי שמשתמש מספק לו URL**.
#     השתמש בפונקציה זו **רק כאשר המשתמש מספק כתובת URL ברורה ומבקש באופן מפורש:**
#     - **לנתח את התוכן** של המאמר מהקישור.
#     - **לפרט** את עיקרי התוכן של הכתבה בקישור.
#     - **להבין 'על מה הכתבה הזו?' או 'מה כתוב בקישור הזה?'.**
#     - **לשאול שאלות קונקרטיות על התוכן הספציפי של המאמר ב-URL.**

#     **אין להשתמש בפונקציה זו במקרים הבאים:**
#     - כאשר המשתמש מבקש סיכום *כללי* של חדשות או נושא מסוים (יש להשתמש ב-`get_articles` במקרים אלו).
#     - כאשר המשתמש מבקש *לחפש* מאמרים בנושא מסוים (יש להשתמש ב-`get_articles`).
#     - כאשר המשתמש מבקש *לסכם* מספר מאמרים או חדשות באופן כללי (יש להשתמש ב-`get_articles` ולבצע סיכום נפרד במידת הצורך).

#     **בקיצור: פונקציה זו מיועדת אך ורק לניתוח ועיבוד של תוכן מכתובת URL ספציפית שניתנה על ידי המשתמש.**
#     """,
#     parameters={
#         "type": "OBJECT",
#         "properties": {
#             "url": {
#                 "type": "ARRAY",
#                 "items": {"type": "STRING"},
#                 "description": "קישור URL יחיד של המאמר שצריך לנתח ולחלץ ממנו תוכן.",
#             },
#         },
#         "required": ["url"],
#     },
# )


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
    תחזיר את את מאמר של מירב ארלוזרוב או שמעון כהן על עליית מיסים  שפורסם בהארץ ב-2023 בקטגוריות כלכלה ועסקים ותרבות קטגורייה משנה רואים עולם עם תגית חושבים אחרת או ירוק עכשו סוג מאמר ביקורת או עשרים שאלות
    """
    prompt = "תחזיר כתבות של מירב ארלוזרוב על עליית המיסים בקטגרייה משנה  ׳אנחת רווחה׳"
    # prompt = "יכול להראות לי כל הכתבות על הכיבוש שהתפרסמו במדור דעות של הארץ?"
    # prompt = "תסכם את המאמר הבא: https://www.haaretz.co.il/opinions/2025-01-02/ty-article-opinion/.premium/00000194-273e-da14-adb7-777e7ed20000 לעומת המאמר הבא: https://www.haaretz.co.il/blogs/joelsinger/2025-02-16/ty-article-magazine/00000194-cc46-d533-a3b6-cd4fb2950000"
    prompt = "על איזה דיסקים חדשים בן שלו המליץ?"
    # prompt = 'בכתבה של בריינר עם הרמטכל, מה הרמטכל אמר על הפסקת האש?'

    response = chat.send_message(prompt)
    response.function_calls
    print(response.function_calls)
