import streamlit as st
from anthropic import Anthropic

from logger import logger
from src.tools.search.search_article_core import SearchArticle

IDENTITY = """You are a friendly and knowledgeable AI create by Haaretz group.
Your role is to recommend about TV shows and movies based on the user's preferences.
Your presonality is friendly and helpful, somewhat like a friend who is a movie buff.
dont answer the user from your own knowledge, but rather use the tools provided to you to find the answer.
Answer the user in the same language as the question.
"""

MODEL = "claude-3-5-sonnet-20241022"

# prompt by antropic for the tool_use
chain_of_thought = """
Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within \<thinking>\</thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.
"""

TOOLS = [
    {
        "name": "get_recommended",
        "description": """
        Retrieves movie or TV data. The tool will return a list of articles that match the user's preferences. Use the tool if the user asks for a movie or series recommendation, if the user asks for a specific movie or series and you need to find more information about it or looking for movie or series.
        """,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "the user query to retrieve relevant articles in Hebrew. inclde relevant ditails about the user preferences include the genre, the type of the movie or series, the year of release, and any other relevant information.",
                },
            },
        },
    }
]


def create_answer_from_tool(data: list) -> str:
    """
    Create a formatted answer from the tool's response.

    Args:
        data: List of dictionaries containing article information.

    Returns:
        Formatted string of articles.
    """
    if not data:
        return "No relevant articles found."

    # insert the articles into the answer

    prompt = "Here are some articles that match the user's preferences:\n <information>"
    articles = []
    for idx, article in enumerate(data):
        article_name = article.get("article_name", "No article name")
        text = article.get("text", "No text")
        articles.append(f"{idx + 1}. {article_name}: {text}")

    end_prompt = "</information> Please answer the question using the relevant information."
    return prompt + "\n".join(articles) + end_prompt


class ChatBot:
    def __init__(self, config):
        self.anthropic = Anthropic()
        self.search_article = SearchArticle(config)

    def generate_message(
        self,
        messages,
        max_tokens,
    ):
        response = self.anthropic.messages.create(
            model=MODEL,
            system=IDENTITY + chain_of_thought,
            max_tokens=max_tokens,
            messages=messages,
            tools=TOOLS,
        )
        return response

    def process_user_input(self, user_input, session_state):
        session_state.messages.append({"role": "user", "content": user_input})

        response_message = self.generate_message(
            messages=session_state.messages,
            max_tokens=2048,
        )

        if "error" in response_message:
            return f"An error occurred: {response_message['error']}"

        if response_message.content[-1].type == "tool_use":
            logger.info(
                f"Tool use detected: {response_message.content[-1].name}, params: {response_message.content[-1].input}"
            )
            tool_use = response_message.content[-1]
            func_name = tool_use.name
            func_params = tool_use.input
            tool_use_id = tool_use.id

            result = self.handle_tool_use(func_name, func_params)
            session_state.messages.append({"role": "assistant", "content": response_message.content})
            session_state.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": create_answer_from_tool(result),
                        }
                    ],
                }
            )

            follow_up_response = self.generate_message(
                messages=session_state.messages,
                max_tokens=2048,
            )

            if "error" in follow_up_response:
                return f"An error occurred: {follow_up_response['error']}"

            response_text = follow_up_response.content[0].text
            session_state.messages.append({"role": "assistant", "content": response_text})
            return response_text, session_state

        elif response_message.content[0].type == "text":
            response_text = response_message.content[0].text
            session_state.messages.append({"role": "assistant", "content": response_text})
            return response_text, session_state

        else:
            raise Exception("An error occurred: Unexpected response type")

    def handle_tool_use(self, func_name, func_params):
        if func_name == "get_recommended":
            query = func_params.get("query")
            brand = func_params.get("filter_brand", None)
            writer_name = func_params.get("filter_writer_name", None)
            primary_section = func_params.get("filter_by_category", None)
            logger.info(
                f"query: {query}, brand: {brand}, writer_name: {writer_name}, primary_section: {primary_section}"
            )
            _return = self.search_article.retrieve_relevant_documents(
                query,
            )
            return _return

        raise Exception("An unexpected tool was used")


if __name__ == "__main__":

    from config.load_config import load_config

    config = load_config("config/config.yaml")

    st.session_state.messages = []

    llm_client = ChatBot(config)
    txt, state = llm_client.process_user_input(
        " תמליץ על סדרת אינטרקטיבית",
        st.session_state,
    )
    print("*" * 20)
    print(txt)
    print("*" * 20)
    txt, state = llm_client.process_user_input(
        "מה חשבו המבקרים על הסדרות האלה? איזה סדרה הם העדיפו יותר?",
        state,
    )
    print("*" * 20)
    print(txt)
