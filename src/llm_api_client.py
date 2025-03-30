import streamlit as st
from anthropic import Anthropic

from config.config import IDENTITY, MODEL, TOOLS
from src.tools.search.search_article_core import SearchArticle
from logger import logger


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
        try:
            response = self.anthropic.messages.create(
                model=MODEL,
                system=IDENTITY,
                max_tokens=max_tokens,
                messages=messages,
                tools=TOOLS,
            )
            return response
        except Exception as e:
            return {"error": str(e)}

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
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            return _return

        raise Exception("An unexpected tool was used")


if __name__ == "__main__":

    from config.load_config import load_config

    config = load_config("config/config.yaml")

    st.session_state.messages = []

    llm_client = ChatBot(st.session_state, config)
    print(llm_client.process_user_input(" תמליץ על סדרת מתח טובה בנטפליקס, שביים יצחק קורניצ'קיו ומשחקים בו ג'וש רדנור וג'יימי פוקס"))