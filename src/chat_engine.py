import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

from config.load_config import load_config
from src.llm_api_client import ChatBot



def main():
    config = load_config("config/config.yaml")

    st.title("הארץ")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "user", "content": "TASK_SPECIFIC_INSTRUCTIONS"},
            {"role": "assistant", "content": "Understood"},
        ]

    chatbot = ChatBot(st.session_state, config)

    # Display user and assistant messages skipping the first two
    for message in st.session_state.messages[2:]:
        # ignore tool use blocks
        if isinstance(message["content"], str):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_msg := st.chat_input("Type your message here..."):
        st.chat_message("user").markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Eva is thinking..."):
                response_placeholder = st.empty()
                full_response = chatbot.process_user_input(user_msg)
                response_placeholder.markdown(full_response)


if __name__ == "__main__":
    main()