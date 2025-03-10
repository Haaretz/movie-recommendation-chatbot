import gradio as gr

from config.load_config import load_config
from logger import logger
from src.chat_engine_utillity.save_chat import (
    get_new_session_timestamp,
    save_chat_history,
    save_feedback,
)
from src.llm_api_client import LLMClient

# from utility.replace_placeholder import replace_placeholder


def reset_chat_and_timestamp():
    llm_api_client.reset_chat_session()
    new_timestamp = get_new_session_timestamp()
    return new_timestamp


def process_message(message, chat_history, session_timestamp, start_date, end_date):
    if message is None or message == "":
        return
    if start_date is None or end_date is None:
        start_date = None
        end_date = None

    formatted_history = []
    if chat_history:
        formatted_history = chat_history

    try:
        response_generator = llm_api_client.send_message(message, start_date, end_date).text

        partial_response = ""
        for chunk in response_generator:
            partial_response += chunk
            updated_history = formatted_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial_response},
            ]
            yield updated_history, "", updated_history

        save_history_for_saving = formatted_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": partial_response},
        ]
        save_chat_history(save_history_for_saving, session_timestamp)
    except Exception as e:
        logger.exception(f"Error in process_message: {e}")
        error_message = "אירעה שגיאה בעיבוד ההודעה."
        updated_history = formatted_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_message},
        ]
        yield updated_history, "", updated_history


if __name__ == "__main__":
    config = load_config("config/config.yaml")
    prompts = load_config("config/prompts.yaml")

    api_key = config["llm"].get("GOOGLE_API_KEY")
    model_name = config["llm"].get("llm_model_name", "gemini-pro")
    logger.info(f"Using model: {model_name}")
    sys_instruct = prompts["system_instructions"]

    # sys_instruct = replace_placeholder(sys_instruct, "{currentDateTime}", datetime.datetime.now().strftime("%Y-%m-%d"))

    llm_api_client = LLMClient(model_name, api_key, sys_instruct, config)

    logger.info("chat interface is starting")

    initial_session_timestamp = get_new_session_timestamp()

    with gr.Blocks(
        css="""
    .gr-examples div { /
      font-size: 105px;
    }
    """
    ) as demo:
        current_chat_history = gr.State([])
        session_timestamp_state = gr.State(initial_session_timestamp)
        feedback_modal_state = gr.State(False)
        custom_date_visibility_state = gr.State(False)
        start_date_state = gr.State(None)
        end_date_state = gr.State(None)
        gr.Markdown(
            """<h1><center>🎩 הברון בוחן העובדות</center></h1>
        <center>המלצות סרטים וסדרות הארץ</center>
        """
        )
        chatbot = gr.Chatbot(height=500, rtl=True, type="messages")
        msg = gr.Textbox(placeholder="הקלד הודעה לברון כאן", label="הודעה לברון", rtl=True)
        clear_button = gr.ClearButton([msg, chatbot, current_chat_history])

        with gr.Row():
            send_btn = gr.Button("שלח הודעה לברון")
            feedback_button = gr.Button("משוב")

        examples = gr.Examples(
            examples=[
                "תסכם לי את התיקים השונים במשפט נתניהו",
                "איזה דיסקים חדשים בן שלו המליץ עליהם השבוע?",
                "תכתוב פסקה על חברת פאראגון",
                "תכתוב שאלות ותשובות על חוק הרבנים",
                "תחזיר המלצות לסרטים שיש סיכוי שיזכו באוסקר",
                """לפי כתבה של יהושע (ג'וש) בריינר, על מה חנמאל דורפמן נחקר במח"ש?""",
                "מהו פער דיגטלי",
                "תסכם את חדשות האקולוגיה מהשבוע האחרון",
                "האם היו דיווחים על רציחות בחברה הערבית השבוע?",
            ],
            inputs=msg,
        )

        with gr.Column(visible=False) as feedback_modal:
            feedback_text_input = gr.Textbox(lines=5, placeholder="הזן את המשוב שלך כאן", label="משוב", rtl=True)
            user_name_input = gr.Textbox(placeholder="השם שלך (אופציונלי)", label="שם", rtl=True)
            submit_feedback_btn = gr.Button("שלח משוב")
            cancel_feedback_btn = gr.Button("ביטול")

            cancel_feedback_btn.click(
                lambda: (gr.update(visible=False), False), outputs=[feedback_modal, feedback_modal_state]
            )

            submit_feedback_btn.click(
                save_feedback,
                inputs=[feedback_text_input, user_name_input, current_chat_history, session_timestamp_state],
                outputs=[feedback_text_input],
            ).success(lambda: (gr.update(visible=False), False), outputs=[feedback_modal, feedback_modal_state])

        feedback_button.click(lambda: (gr.update(visible=True), True), outputs=[feedback_modal, feedback_modal_state])

        def add_user_message(user_message, history):
            history = history + [{"role": "user", "content": user_message}]
            return history, history

        def bot_response(history, session_timestamp, start_date, end_date):
            try:
                user_message = history[-1]["content"]
                full_history = history
                for response in process_message(
                    user_message, full_history[:-1], session_timestamp, start_date, end_date
                ):
                    yield response
            except Exception as e:
                logger.info(f"Error in bot_response: {e}")
                save_chat_history(full_history, "Error - " + session_timestamp)
                error_message = "אירעה שגיאה בתגובת הבוט. אנא נסה שוב מאוחר יותר"
                updated_history = history + [{"role": "assistant", "content": error_message}]
                yield updated_history, "", updated_history

        msg.submit(add_user_message, [msg, current_chat_history], [chatbot, current_chat_history]).then(
            bot_response,
            [current_chat_history, session_timestamp_state, start_date_state, end_date_state],
            [chatbot, msg, current_chat_history],
        )
        send_btn.click(add_user_message, [msg, current_chat_history], [chatbot, current_chat_history]).then(
            bot_response,
            [current_chat_history, session_timestamp_state, start_date_state, end_date_state],
            [chatbot, msg, current_chat_history],
        )

        clear_button.click(reset_chat_and_timestamp, outputs=[session_timestamp_state])


    try:
        iface = demo.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=8080,
        )
    except Exception as e:
        logger.info(f"Error launching Gradio demo: {e}")
        print("Failed to launch the Gradio interface. Please check the logs.")
