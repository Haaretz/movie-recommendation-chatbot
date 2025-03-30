
IDENTITY = """You are a friendly and knowledgeable AI create by Haaretz group.
Your role is to recommend about TV shows and movies based on the user's preferences.
"""

MODEL = "claude-3-haiku-20240307"

# prompt by antropic for the tool_use
chain_of_thought = """ 
Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within \<thinking>\</thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.
"""

TOOLS = [
    {
        "name": "get_recommended",
        "description": "Provide movie or TV show recommendations based on user preferences. The tool will return a list of articles that match the user's preferences. Use the tool if the user asks for a movie or series recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the user query to retrieve relevant articles"},
                "genre": {
                    "type": "string",
                    "description": "filter by genre",
                    },
                "type": {
                    "type": "string",
                    "description": "Type of content: Series or Movie",
                    "enum": ["סדרה", "סרט"],
                },
                "director": {"type": "string", "description": "filter by director"},
                "producer": {"type": "string", "description": "filter by producer"},
                "actors": {"type": "string", "description": "filter by actors"},
                "distribution_platform": {
                    "type": "string",
                    "description": "filter by distribution platform",
                    "enum": ["Netflix", "Yes", "Hot", "Apple TV", "סטינג", "סלקום TV", "פרטנר TV", "ספקטרום"],
                },
                "movie_length": {
                    "type": "integer",
                    "description": "filter by movie length",
                },
                "language": {
                    "type": "string",
                    "description": "filter by language",
                    "enum": ["עברית", "אנגלית"],
                },
                "release_Year": {
                    "type": "integer",
                    "description": "filter by release year",
                },
                "rating": {
                    "type": "number",
                    "description": "filter by rating",
                },

            },
        },
    }
]
