import streamlit as st
from anthropic import Anthropic

from logger import logger
from src.tools.search.search_article_core import SearchArticle

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
            },
        },
    }
]


class ChatBot:
    def __init__(self, session_state, config):
        self.anthropic = Anthropic()
        self.session_state = session_state
        self.search_article = SearchArticle(config)

    def generate_message(
        self,
        messages,
        max_tokens,
    ):
        response = self.anthropic.messages.create(
            model=MODEL,
            system=IDENTITY,
            max_tokens=max_tokens,
            messages=messages,
            tools=TOOLS,
        )
        return response

    def process_user_input(self, user_input):
        self.session_state.messages.append({"role": "user", "content": user_input})

        response_message = self.generate_message(
            messages=self.session_state.messages,
            max_tokens=2048,
        )

        if "error" in response_message:
            return f"An error occurred: {response_message['error']}"

        if response_message.content[-1].type == "tool_use":
            tool_use = response_message.content[-1]
            func_name = tool_use.name
            func_params = tool_use.input
            tool_use_id = tool_use.id

            result = self.handle_tool_use(func_name, func_params)
            self.session_state.messages.append({"role": "assistant", "content": response_message.content})
            self.session_state.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "tool_use_id": tool_use_id,
                            "source": {"type": "content", "id": "retrieved_articles"},
                            "content": result,
                            "context": "retrived articles from the RAG model",
                            "citations": {"enabled": True},
                        }
                    ],
                }
            )

            follow_up_response = self.generate_message(
                messages=self.session_state.messages,
                max_tokens=2048,
            )

            if "error" in follow_up_response:
                return f"An error occurred: {follow_up_response['error']}"

            response_text = follow_up_response.content[0].text
            self.session_state.messages.append({"role": "assistant", "content": response_text})
            return response_text

        elif response_message.content[0].type == "text":
            response_text = response_message.content[0].text
            self.session_state.messages.append({"role": "assistant", "content": response_text})
            return response_text

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

    llm_client = ChatBot(st.session_state, config)
    print(
        llm_client.process_user_input(
            " תמליץ על סדרת מתח טובה בנטפליקס, שביים יצחק קורניצ'קיו ומשחקים בו ג'וש רדנור וג'יימי פוקס"
        )
    )
