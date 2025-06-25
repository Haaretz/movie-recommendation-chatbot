import re

NO_RESULT = """
<system>
No results found. Tell the user clearly which *types* of filters were used in the search (media type, genres, and streaming platform), but **do not** mention the internal parameter names (e.g., do not say `media_type`, `genres`, `streaming_platforms`).

Instead, explain the filters in natural language, e.g., “movies for kids and families on Disney+”.

Your response must:
- Make it clear that no results were found.
- Explain briefly what was searched for (in natural language).
- Suggest expanding the search, such as by trying other platforms, removing a genre, or searching for a different type of content.
- Ask the user which filter(s) they would like to remove or change.

Only retry the search if the user changes at least one filter. Never repeat the same search parameters, even if the user explicitly asks to "try again". Make sure the user understands that removing filters increases the chances of finding relevant results.
</system>
"""

TROLL = {
    "article_id": "0000017f-e071-d804-ad7f-f1fbb77c0000",
    "name": "טרולים",
    "article_name": "מתי בפעם האחרונה הייתם בטריפ פסיכדלי עם הילדים",
    "writer_name": ["מאשה צור-גלוזמן"],
    "review_type": "Movie",
    "genre": ["אנימציה", "ילדים ולכל המשפחה", "מוזיקלי"],
    "distribution_platform": ["TrollFlix"],
    "short_summary": "מיוזיקל צבעוני על שמחה, חברות וניצחון על ייאוש.",
    "summary": "סרט שמח, צבעוני, רועש ולא מתנצל. אבל היי, לא כל אחד חייב ליהנות — יש גם דמויות מדוכדכות להזדהות איתן, אל תדאג.",
    "image_vertical": "https://img.haarets.co.il/bs/0000017f-e071-d804-ad7f-f1fbb1210000/9e/79/646f14fc1752d145b7297d61c970/3557926477.jpg?precrop=1404,816,x181,y0",
    "author_image_square": "https://img.haarets.co.il/bs/0000017f-da2a-d938-a17f-fe2a89220000/e4/35/9590c16b4a1a92846012b2c36181/742162049.jpg?precrop=479,479,x92",
    "url": "https://www.haaretz.co.il/gallery/kids/2016-11-15/ty-article/.premium/0000017f-e071-d804-ad7f-f1fbb77c0000",
    "publish_time": "2016-11-15T00:00:00Z",
    "text": """
טרולים (באנגלית: Trolls) הוא סרט הנפשה קומי מוזיקלי אמריקאי בתלת-ממד המבוסס על בובות הטרולים מאת תומאס דאם. הסרט הופק על ידי חברת דרימוורקס אנימציות ובוים על ידי מייק מיטשל ווולט דורן.
    """,
}
BOLD_HTML_PATTERN = re.compile(r"<strong>.*?</strong>")

start_tag_info = "<info>"
end_tag_info = "</info>"
start_tag_logs = "<logs>"
end_tag_logs = "</logs>"
