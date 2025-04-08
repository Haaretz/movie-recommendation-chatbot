import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import redis
from google import genai
from google.genai import types
from google.genai.types import Content, FunctionCall, FunctionResponse, Part

from constant import NO_RESULT
from logger import logger
from src.tools.import_tools import qdrant_tools
from src.tools.search.search_article_core import SearchArticle


class LLMClient:
    def __init__(self, model_name: str, api_key: str, sys_instruct: str, config):
        """
        Initialize the LLMApiClient using the chat conversation API.

        Args:
            model_name (str): The name of the language model to use (e.g., "gemini-pro", "gemini-2.0-flash").
            api_key (str): The API key for accessing the LLM service.
            sys_instruct (str, optional): System instructions (not directly used in this chat API example). Defaults to None.
        """
        self.search_article = SearchArticle(config)
        self.sys_instruct = sys_instruct
        self.api_key = api_key
        self.model_name = model_name
        self.filed_for_frontend = config.get("filed_for_frontend", {})
        try:
            self.chat_session = self._initialize_client(self.sys_instruct, self.api_key, self.model_name)
            logger.info(f"LLMClient initialized for model: {model_name}")
        except Exception as e:
            logger.info(f"Error initializing LLMClient: {e}")
            self.chat_session = None  # Handle initialization failure

        # --- Redis Initialization ---
        redis_url = os.getenv("REDIS_URL")
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.redis_client.ping()
        logger.info("Successfully connected to Redis.")

    def _initialize_client(self, sys_instruct, api_key, model_name):
        """
        Initializes the Google Generative AI client.
        """
        try:
            self.client = genai.Client(vertexai=False, api_key=api_key)
            chat = self.client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruct,
                    tools=[qdrant_tools],
                ),
            )
            logger.debug("Successfully initialized genai client.")
            return chat
        except Exception as e:
            logger.info(f"Error initializing genai client: {e}")
            raise  # Re-raise the exception to be handled in the caller (init)

    def _create_chat_session(self, history: Optional[List[Content]] = None) -> Optional[genai.ChatSession]:
        """
        Creates a new chat session, optionally initialized with provided history.
        Uses the base client initialized in __init__.
        """
        if not self.client:
            logger.error("Base GenerativeModel client not initialized. Cannot create chat session.")
            return None
        try:
            # Start chat using the already configured GenerativeModel instance
            chat = self.client.start_chat(
                history=history if history else [], enable_automatic_function_calling=True  # Enable automatic calling
            )
            logger.debug("Successfully created new chat session.")
            return chat
        except Exception as e:
            logger.error(f"Error creating chat session: {e}", exc_info=True)
            return None  # Return None on failure

    def _translate_english_query(self, query: str):
        is_hebrew = any("\u0590" <= char <= "\u05FF" or "\uFB1D" <= char <= "\uFB4F" for char in query)
        if is_hebrew:
            logger.debug("Query is likely Hebrew, returning original query.")
            return query

        translation_prompt = f"Translate the following English query to Hebrew: '{query}'. Return only the translation."
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=translation_prompt,
        )
        translated_query = response.text.strip()
        logger.debug(f"Translated query to Hebrew: '{translated_query}'")
        return translated_query

    def _filter_fields_and_call_tool(self, function_calls):
        """
        Processes detected function calls, executes the tool, and returns response parts.

        Args:
            function_calls (list): List of FunctionCall objects from the LLM response.

        Returns:
            list: A list of Part objects containing function responses.
                  Returns an empty list if no relevant function calls are found or errors occur.
        """
        parts = []
        processed_call_names = set()

        for call in function_calls:
            call_name = call.name
            call_args = call.args

            logger.info(f"Processing function call: {call_name} with args: {call_args}")

            if call_name == "recommendations_for_tv_and_movies":
                query = call_args.get("query")
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
                    search_results = self.search_article.retrieve_relevant_documents(translated_query)
                    if len(search_results) == 0:
                        search_results = NO_RESULT
                        return search_results, ""
                    logger.info(
                        f"Tool '{call_name}' executed successfully for query: '{translated_query}'. Results obtained."
                    )
                    # logger.debug(f"Search results: {search_results}")
                    # Ensure results are serializable (e.g., string or basic dict/list)

                    # Create the function response Part
                    parts.append(
                        Part.from_function_response(
                            name=call_name,
                            response={
                                "content": search_results,
                            },
                        )
                    )
                    processed_call_names.add(call_name)  # Mark as processed

                except Exception as e:
                    logger.error(f"Error executing tool for function call {call_name}: {e}", exc_info=True)
                    # Optionally append an error response part
                    parts.append(
                        Part.from_function_response(
                            name=call_name,
                            response={
                                "content": f"שגיאה בביצוע החיפוש עבור: '{translated_query}'. {str(e)}",
                            },
                        )
                    )
                    processed_call_names.add(call_name)  # Mark as processed even if error occurred

            else:
                logger.warning(f"Received unhandled function call: {call_name}")
                # Optionally handle other function calls or ignore them

        metadata = self.metadata_for_backend(search_results)

        logger.info(f"Generated {len(parts)} function response parts.")
        return parts, metadata

    def metadata_for_backend(self, metadata):

        # metadata = [s.model_dump()['payload'] for s in metadata]
        metadata = [{key: item[key] for key in self.filed_for_frontend} for item in metadata]
        return json.dumps(metadata)

    def _load_history_from_redis(self, user_id: str) -> List[Content]:
        key = f"chat_history:{user_id}"
        json_history = self.redis_client.get(key)
        if json_history:
            logger.debug(f"Loaded history for user {user_id} from Redis.")
            return self._deserialize_history(json_history)
        else:
            logger.debug(f"No history found for user {user_id} in Redis.")
            return []

    def _serialize_history(self, history: List[Content]) -> str:
        """Serializes chat history (list of Content objects) to a JSON string."""
        serializable_history = []
        for content in history:
            serializable_parts = [self._serialize_part(part) for part in content.parts if part]  # Filter empty parts
            if serializable_parts:  # Only add if there are valid parts
                serializable_history.append({"role": content.role, "parts": serializable_parts})
        return json.dumps(serializable_history)

    def _serialize_part(self, part: Part) -> Dict[str, Any]:
        """Serializes a Part object to a dictionary."""
        if part.text:
            return {"text": part.text}
        elif part.function_call:
            return {"function_call": {"name": part.function_call.name, "args": dict(part.function_call.args)}}
        elif part.function_response:
            return {
                "function_response": {
                    "name": part.function_response.name,
                    "response": dict(part.function_response.response),
                }
            }
        else:
            # Handle potential other part types or empty parts
            logger.warning("Serializing an unknown or empty Part type.")
            return {}

    def _deserialize_history(self, json_string: str) -> List[Content]:
        """Deserializes a JSON string back into chat history (list of Content objects)."""
        history_data = json.loads(json_string)
        history = []
        for item in history_data:
            role = item.get("role", "user")
            if not role:
                logger.error(f"History item missing role: {item}")
                continue  # Skip malformed item

            parts_data = item.get("parts", [])
            deserialized_parts = []
            for part_data in parts_data:
                part = self._deserialize_part(part_data)
                if part:  # Only add successfully deserialized parts
                    deserialized_parts.append(part)

                if deserialized_parts:  # Only add Content if it has parts
                    history.append(Content(role=role, parts=deserialized_parts))
                else:
                    logger.warning(f"Skipping history item with no valid parts after deserialization: {item}")

        return history

    def _deserialize_part(self, data: Dict[str, Any]) -> Optional[Part]:
        """Deserializes a dictionary back to a Part object."""
        if "text" in data:
            return Part(text=data["text"])
        elif "function_call" in data:
            fc_data = data["function_call"]
            return Part(function_call=FunctionCall(name=fc_data["name"], args=fc_data["args"]))
        elif "function_response" in data:
            fr_data = data["function_response"]
            # Use Part.from_function_response helper if available and appropriate,
            # otherwise construct manually or adjust based on library specifics.
            # Assuming direct construction works or response is simple dict:
            return Part(function_response=FunctionResponse(name=fr_data["name"], response=fr_data["response"]))
            # Alternative using helper (might need adjustments):
            # return Part.from_function_response(name=fr_data["name"], response=fr_data["response"])
        else:
            logger.warning(f"Deserializing an unknown Part structure: {data}")
            return None

    async def streaming_message(self, message: str, user_id: str) -> AsyncGenerator[str, None]:
        """
        Sends message, handles streaming and function calls. Minimal error checks.
        """
        # loaded_history = self._load_history_from_redis(user_id)

        stream = self.chat_session.send_message_stream(message)

        collected_function_calls: List[FunctionCall] = []
        current_turn_involved_function_call = False

        # --- Phase 1: Consume initial stream, yield text, collect calls ---
        # logger.debug("Starting Phase 1: Consuming initial model stream...")
        for chunk in stream:
            if chunk.text:
                yield chunk.text
            if chunk.candidates[0].content.parts[0].function_call:
                collected_function_calls.append(chunk.candidates[0].content.parts[0].function_call)
                current_turn_involved_function_call = True

        # --- Phase 2 & 3: Process calls, send responses, stream final answer ---
        if current_turn_involved_function_call:

            function_response_parts, metadata = self._filter_fields_and_call_tool(collected_function_calls)
            yield metadata

            response_stream_after_fc = self.chat_session.send_message_stream(
                function_response_parts,
            )
            for final_chunk in response_stream_after_fc:
                if final_chunk.text:
                    yield final_chunk.text


async def main_cli():
    from config.load_config import load_config

    prompts = load_config("config/prompts.yaml")

    sys_instruct = prompts.get("system_instructions")

    config = load_config("config/config.yaml")

    llm_config = config.get("llm", {})
    api_key = llm_config.get("GOOGLE_API_KEY")
    model_name = llm_config.get("llm_model_name")

    llm_client = LLMClient(model_name=model_name, api_key=api_key, sys_instruct=sys_instruct, config=config)
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
        async for chunk in llm_client.streaming_message(user_message):
            print(chunk, end="", flush=True)
            if chunk:
                full_response += chunk
        print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_cli())
