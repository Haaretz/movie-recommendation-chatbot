import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import redis
from google import genai
from google.genai import types
from google.genai.types import Content, FunctionCall, Part

from constant import NO_RESULT
from logger import logger
from src.tools.import_tools import qdrant_tools
from src.tools.search.search_article_core import SearchArticle


class LLMClient:
    def __init__(self, model_name: str, api_key: str, sys_instruct: str, config):
        """
        Initialize the LLMApiClient using the chat conversation API.

        Args:
            model_name (str): The language model to use (e.g., "gemini-pro", "gemini-2.0-flash").
            api_key (str): The API key for accessing the LLM service.
            sys_instruct (str): System instructions.
            config: Configuration data.
        """
        self.search_article = SearchArticle(config)
        self.sys_instruct = sys_instruct
        self.api_key = api_key
        self.model_name = model_name
        self.filed_for_frontend = config.get("filed_for_frontend", {})
        self.filed_for_llm = config.get("filed_for_llm", {})

        # --- LLM Initialization ---
        self.client = genai.Client(vertexai=False, api_key=api_key)

        # --- Redis Initialization ---
        redis_url = os.getenv("REDIS_URL")
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.redis_client.ping()
        logger.info("Successfully connected to Redis.")

    def _create_chat_session(self, history: Optional[List[Content]] = None):
        """
        Creates a new chat session with the provided history.
        """
        chat = self.client.chats.create(
            model=self.model_name,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=self.sys_instruct,
                tools=[qdrant_tools],
            ),
        )
        logger.debug("Successfully created new chat session.")
        return chat

    def _translate_english_query(self, query: str):
        """
        Translates an English query to Hebrew if needed.
        """
        is_hebrew = any("\u0590" <= char <= "\u05FF" or "\uFB1D" <= char <= "\uFB4F" for char in query)
        if is_hebrew:
            logger.debug("Query is likely Hebrew, returning original query.")
            return query

        translation_prompt = f"Translate the following English query to Hebrew: '{query}'. Return only the translation."
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=translation_prompt,
            config=genai.types.GenerateContentConfig(thinking_config=genai.types.ThinkingConfig(thinking_budget=1024)),
        )
        translated_query = response.text.strip()
        logger.debug(f"Translated query to Hebrew: '{translated_query}'")
        return translated_query

    def _filter_fields_and_call_tool(self, function_calls):
        """
        Processes detected function calls, executes the tool, and returns response parts.
        Also saves function call information to Redis.
        """
        parts = []
        processed_call_names = set()

        for call in function_calls:
            call_name = call.name
            call_args = call.args

            logger.info(f"Processing function call: {call_name} with args: {call_args}")

            if call_name == "get_dataset_articles":
                query = call_args.get("query")
                streaming: str = call_args.get("streaming_platforms", [])
                genres: str = call_args.get("Genres", [])
                if not query:
                    logger.error(f"Missing 'query' argument for function call {call_name}")
                    parts.append(
                        Part.from_function_response(
                            name=call_name,
                            response={"error": f"שגיאה: לא סופק 'query' לקריאה לפונקציה {call_name}."},
                        )
                    )
                else:
                    translated_query = self._translate_english_query(query)
                    try:
                        search_results = self.search_article.retrieve_relevant_documents(
                            translated_query, streaming, genres
                        )
                        if len(search_results) == 0:
                            search_results = NO_RESULT
                            return search_results, ""
                        logger.info(
                            f"Tool '{call_name}' executed successfully for query: '{translated_query}'. Results obtained."
                        )
                        parts.append(
                            Part.from_function_response(
                                name=call_name,
                                response={
                                    "content": [
                                        {key: item[key] for key in self.filed_for_llm} for item in search_results
                                    ],
                                },
                            )
                        )
                        processed_call_names.add(call_name)
                        # Save the function call information into Redis
                        self._save_function_call_to_redis(
                            "123", {"name": call_name, "args": call_args, "translated_query": translated_query}
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool for function call {call_name}: {e}", exc_info=True)
                        parts.append(
                            Part.from_function_response(
                                name=call_name,
                                response={
                                    "content": f"שגיאה בביצוע החיפוש עבור: '{translated_query}'. {str(e)}",
                                },
                            )
                        )
                        processed_call_names.add(call_name)
            else:
                logger.warning(f"Received unhandled function call: {call_name}")

        metadata = self.metadata_for_backend(search_results)
        logger.info(f"Generated {len(parts)} function response parts.")
        return parts, metadata, search_results

    def metadata_for_backend(self, metadata):
        # Filter metadata fields as defined in the configuration.
        metadata = [{key: item.get(key, None) for key in self.filed_for_frontend} for item in metadata]
        return json.dumps(metadata, ensure_ascii=False)

    def _load_history_from_redis(self, user_id: str) -> List[Content]:
        """
        Loads the chat history for a given user from Redis.
        Uses a Redis list where each element is a JSON-encoded message.
        """
        key = f"chat_history:{user_id}"
        # Retrieve all elements from the list
        json_history_list = self.redis_client.lrange(key, 0, -1)
        if json_history_list:
            logger.debug(f"Loaded history for user {user_id} from Redis.")
            # Deserialize each message into a Content object
            return [self._deserialize_message(message_str) for message_str in json_history_list]
        else:
            logger.debug(f"No history found for user {user_id} in Redis.")
            return []

    def _save_message_to_redis(self, user_id: str, message: Content) -> None:
        """
        Saves a single chat message to Redis as a JSON-encoded string.
        Appends the new message to the end of the Redis list.
        """
        key = f"chat_history:{user_id}"
        serialized = self._serialize_message(message)
        self.redis_client.rpush(key, serialized)

    def _save_function_call_to_redis(self, user_id: str, function_call: Dict[str, Any]) -> None:
        """
        Saves function call information to Redis.
        The function call is stored as a JSON message with role 'function'.
        """
        key = f"chat_history:{user_id}"
        serialized = json.dumps({"role": "function", "function_call": function_call})
        self.redis_client.rpush(key, serialized)

    def _serialize_message(self, message: Content) -> str:
        """
        Serializes a Content object into a JSON string.
        Only the 'role' and the text of each 'part' are stored.
        """
        return json.dumps(
            {
                "role": message.role,
                "parts": [part.text for part in message.parts],
            }
        )

    def _deserialize_message(self, message_str: str) -> Content:
        """
        Deserializes a JSON string into a Content object.
        """
        data = json.loads(message_str)
        parts = [Part(text=part_text) for part_text in data.get("parts", [])]
        return Content(role=data["role"], parts=parts)

    async def streaming_message(self, message: str, user_id: str) -> AsyncGenerator[str, None]:
        """
        Sends a message, handles streaming response, processes function calls, and saves updated history in Redis.
        """
        collected_function_calls: List[FunctionCall] = []
        current_turn_involved_function_call = False

        # 1. Load history from Redis (each message is stored separately)
        loaded_history = self._load_history_from_redis(user_id)

        # 2. Create a new chat session with the loaded history
        current_chat_session = self._create_chat_session(history=loaded_history)

        full_response_text = ""  # To collect full text response from the assistant

        # 3. Send the message using the chat session streaming
        stream = current_chat_session.send_message_stream(message)

        for chunk in stream:
            if chunk.text:
                yield chunk.text
                full_response_text += chunk.text
            if (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
                and isinstance(chunk.candidates[0].content.parts, list)
                and len(chunk.candidates[0].content.parts) > 0
                and chunk.candidates[0].content.parts[0]
                and hasattr(chunk.candidates[0].content.parts[0], "function_call")
                and chunk.candidates[0].content.parts[0].function_call
            ):
                collected_function_calls.append(chunk.candidates[0].content.parts[0].function_call)
                current_turn_involved_function_call = True

        if current_turn_involved_function_call:
            # Process the function call: execute tool and update with metadata
            function_response_parts, metadata, search_results = self._filter_fields_and_call_tool(
                collected_function_calls
            )
            yield metadata

            response_stream_after_fc = current_chat_session.send_message_stream(function_response_parts)
            try:
                for final_chunk in response_stream_after_fc:
                    if final_chunk.text:
                        yield final_chunk.text
                        full_response_text += final_chunk.text  # TODO: consider if need to save the output seaprately
            except Exception as e:
                logger.error(f"Error during function call response streaming: {e}", exc_info=True)
                yield f"Error during function call response streaming: {e}"
        # 4. Create Content objects for the user message and the assistant's response
        user_content = Content(role="user", parts=[Part(text=message)])
        self._save_message_to_redis(user_id, user_content)

        if current_turn_involved_function_call:
            logger.info(f"Function call detected. Collected function calls: {collected_function_calls}")
            # Create an assistant message with embedded function call info
            assistant_content = Content(
                role="model", parts=[Part(text=full_response_text, function_call=collected_function_calls[0])]
            )
            search_results = json.dumps(search_results, ensure_ascii=False)
            function_output_content = Content(role="model", parts=[Part(text=search_results)])
            self._save_message_to_redis(user_id, function_output_content)
            self._save_message_to_redis(user_id, assistant_content)
        else:
            assistant_content = Content(role="model", parts=[Part(text=full_response_text)])
            self._save_message_to_redis(user_id, assistant_content)


async def main_cli():
    import numpy as np

    from config.load_config import load_config

    prompts = load_config("config/prompts.yaml")
    sys_instruct = prompts.get("system_instructions")
    config = load_config("config/config.yaml")
    llm_config = config.get("llm", {})
    api_key = llm_config.get("GOOGLE_API_KEY")
    model_name = llm_config.get("llm_model_name")

    llm_client = LLMClient(model_name=model_name, api_key=api_key, sys_instruct=sys_instruct, config=config)
    counter = np.random.randint(1, 99999)
    # counter = 43
    while True:
        user_message = input("You: ")
        if user_message.lower() == "quit":
            break
        if user_message.lower() == "reset":
            try:
                llm_client.reset_chat_session()
                print("--- Chat session reset ---")
            except Exception as e:
                print(f"\nError resetting chat session: {e}")
                logger.error(f"Error during chat session reset: {e}", exc_info=True)
            continue

        print("LLM: ", end="", flush=True)
        full_response = ""

        async for chunk in llm_client.streaming_message(user_message, user_id=counter):
            print(chunk, end="", flush=True)
            if chunk:
                full_response += chunk
        print()


# async def run_test():
#     first_question_article = [
#         {
#             "article_id": "00000194-25d6-dcc4-a1d7-3df6703c0000",
#             "question": "אני רוצה לקרוא ביקורת על הסרט החדש של לוקה גואדנינו, זה שמבוסס על ספר של בורוז על הומו שחי במקסיקו בשנות ה-50 ומחפש גברים צעירים.",
#         },
#         {
#             "article_id": "00000194-2618-dd68-a3be-e6fc06680000",
#             "question": 'אני מחפש ביקורת על הסרט החדש עם ניקול קידמן, זה שהיא משחקת בו מנכ"לית חברת רובוטיקה שיש לה רומן עם מתמחה צעיר. הסרט הוקרן בפסטיבל ונציה, איך הוא?.',
#         },
#         {
#             "article_id": "00000194-2681-ddb6-afdd-77e781220000",
#             "question": "אני מחפש ביקורת על הסרט החדש של עמוס גיתאי, זה שעוסק בשאלה למה יש מלחמות ומשתמש במכתבים של איינשטיין ופרויד.",
#         },
#         {
#             "article_id": "00000194-2b97-d9c2-a79e-2bd7330c0000",
#             "question": 'אפשר בבקשה ביקורת על הסרט החדש "נוספרטו" של רוברט אגרס? אני רוצה לדעת אם הוא נאמן למקורות של סרטי הערפדים הקלאסיים ואיך השחקנים הראשיים, במיוחד לילי-רוז דפ, משחקים?',
#         },
#         {
#             "article_id": "00000194-319b-d555-abbc-b1dfdf250000",
#             "question": '"אני מחפש ביקורת על סרט חדש שמוקרן בפסטיבל חיפה, על אישה אלכוהוליסטית בשם רונה שחוזרת לאי הולדתה בסקוטלנד כדי להתמודד עם ההתמכרות שלה. מישהי כתבה על זה ביקורת?',
#         },
#         {
#             "article_id": "00000194-31db-ddaf-adb7-7bfbaf200000",
#             "question": "אני מחפש סרט איראני חדש שמתרחש בטהרן ועוסק במשפחה שאבא שלה שופט ורואים בו גם קטעים מההפגנות שם. ראיתי שהקרינו אותו בפסטיבל חיפה",
#         },
#         {
#             "article_id": "00000194-34df-d39d-a196-b7ff86ce0000",
#             "question": '"אני מחפש סדרת אנימציה חדשה בנטפליקס, משהו בסגנון באפי קוטלת הערפדים אבל עם מיתולוגיה סינית. שמעתי שיש סדרה על נערה סינית-אמריקאית שנלחמת בשדים, מישהו מכיר?',
#         },
#         {
#             "article_id": "00000194-356b-de88-a3dc-75ef26460000",
#             "question": "אוקיי, שמעתי על סרט חדש של שבי גביזון עם ריצ'רד גיר על אבא שמגלה שהיה לו בן שנהרג. איך קוראים לסרט ומה הוא מספר עליו?",
#         },
#     ]
#     from config.load_config import load_config

#     prompts = load_config("config/prompts.yaml")
#     sys_instruct = prompts.get("system_instructions")
#     config = load_config("config/config.yaml")
#     llm_config = config.get("llm", {})
#     api_key = llm_config.get("GOOGLE_API_KEY")
#     model_name = llm_config.get("llm_model_name")

#     result = []
#     import numpy as np

#     counter = np.random.randint(1, 99999)

#     llm_client = LLMClient(model_name=model_name, api_key=api_key, sys_instruct=sys_instruct, config=config)
#     for q in first_question_article:
#         article_id = q.get("article_id")
#         question = q.get("question")
#         print(f"Article ID: {article_id}")
#         print(f"Question: {question}")
#         full_response = ""
#         async for chunk in llm_client.streaming_message(question, f"{counter}"):
#             full_response += chunk
#             print(chunk, end="", flush=True)
#         counter += 1

#         result.append({"article_id": article_id, "question": question, "answer": full_response})


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_cli())
    # asyncio.run(run_test())
