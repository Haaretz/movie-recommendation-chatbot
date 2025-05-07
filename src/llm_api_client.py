import json
import time
from typing import AsyncGenerator, List, Optional

from google import genai
from google.genai import types
from google.genai.types import Content, FunctionCall, Part

from constant import NO_RESULT, TROLL
from logger import logger
from src.redis_chat_history import RedisChatHistory
from src.tools.import_tools import qdrant_tools
from src.tools.search.search_article_core import SearchArticle


class LLMClient:
    """High‑level wrapper around Gemini chat + article search with streaming I/O."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        sys_instruct: str,
        config,
        redis_store: RedisChatHistory,
    ):
        self.search_article = SearchArticle(config)
        self.sys_instruct = sys_instruct
        self.model_name = model_name
        self.fields_for_frontend = config.get("fields_for_frontend", {})
        self.fields_for_llm = config.get("fields_for_llm", {})
        self.redis = redis_store  # composition – dependency injection

        # --- LLM Initialization ---
        self.client = genai.Client(vertexai=False, api_key=api_key)

    def _create_chat_session(self, history: Optional[List[Content]] = None):
        return self.client.chats.create(
            model=self.model_name,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=self.sys_instruct,
                tools=[qdrant_tools],
            ),
        )

    def _translate_english_query(self, query: str):
        is_hebrew = any("\u0590" <= ch <= "\u05FF" or "\uFB1D" <= ch <= "\uFB4F" for ch in query)
        if is_hebrew:
            return query

        prompt = f"Translate the following English query to Hebrew: '{query}'. Return only the translation."
        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        translated = resp.text.strip()
        logger.debug("Translated query to Hebrew: '%s'", translated)
        return translated

    def _filter_fields_and_call_tool(self, function_calls):
        """
        Process function calls, route to appropriate handlers, and return
        the response parts, frontend metadata, and raw search results.
        """
        parts = []
        search_results = None

        for call in function_calls:
            if call.name == "get_dataset_articles":
                handler_parts, search_results = self._handle_get_dataset_articles(call)
                parts.extend(handler_parts)
            elif call.name == "trigger_troll_response":
                handler_parts, search_results = self._handle_trigger_troll_response(call)
                parts.extend(handler_parts)
            else:
                logger.warning(f"No handler registered for function '{call.name}'")

        # Prepare metadata for frontend if we have list results
        metadata = None
        if isinstance(search_results, list):
            metadata = json.dumps(
                [{k: item.get(k, None) for k in self.fields_for_frontend} for item in search_results],
                ensure_ascii=False,
            )

        return parts, metadata, search_results

    def _handle_get_dataset_articles(self, call):
        """
        Handle 'get_dataset_articles' calls by performing the search and
        formatting the response part.
        """
        args = call.args
        query = args.get("query")
        streaming = args.get("streaming_platforms", [])
        genres = args.get("Genres", [])
        media_type = args.get("media_type")

        logger.info(
            "Executing get_dataset_articles with query=%s, streaming=%s, genres=%s, media_type=%s",
            query,
            streaming,
            genres,
            media_type,
        )

        translated_query = self._translate_english_query(query)
        search_results = self.search_article.retrieve_relevant_documents(
            translated_query, streaming, genres, media_type
        )

        if not search_results:
            # No results found, return a placeholder response
            part = Part.from_function_response(
                name=call.name,
                response={"content": [f"No results found for the query: {translated_query}"]},
            )
            return [part], NO_RESULT

        # Convert results for LLM
        content_list = [{k: item.get(k) for k in self.fields_for_llm} for item in search_results]
        part = Part.from_function_response(name=call.name, response={"content": content_list})
        return [part], search_results

    def _handle_trigger_troll_response(self, call):
        """
        Handle 'trigger_troll_response' calls by returning predefined troll content.
        """
        logger.info("Triggering troll response")

        troll_results = [TROLL]
        content_list = [{k: item.get(k) for k in self.fields_for_llm} for item in troll_results]
        part = Part.from_function_response(name=call.name, response={"content": content_list})
        return [part], troll_results

    def num_tokens(
        self,
        history: List[Content],
        new_user_msg: str,
        assistant_reply: str,
    ) -> tuple[int, int]:
        """
        Return (input_tokens, output_tokens) for the *entire* turn.
        Includes: prior history  + current user   + optional function‑call parts.
        Output is the assistant reply text only (function‑call output is
        already counted on the *input* side because it goes *into* the model).

        All counting is done with the model's tokenizer via
        `client.models.count_tokens`.
        """

        def _count(text: str) -> int:
            return self.client.models.count_tokens(model=self.model_name, contents=text).total_tokens

        # --- INPUT ---
        total_in = 0
        for content in history:
            if content.role in {"user", "model"}:
                if content.parts[0].text:
                    total_in += _count(content.parts[0].text)
                elif content.parts[0].function_response:
                    total_in += _count(json.dumps(content.parts[0].function_response.response))

        total_in += _count(new_user_msg)

        # --- OUTPUT ---
        total_out = _count(assistant_reply)

        return total_in, total_out

    async def streaming_message(self, message: str, user_id: str) -> AsyncGenerator[str, None]:
        collected_calls: List[FunctionCall] = []
        involved_fc = False

        history = self.redis.load_history(user_id)
        chat = self._create_chat_session(history=history)
        full_reply = ""

        total_start = time.time()

        # --- send user message to LLM, collect function calls if any ---
        llm_start = time.time()
        for chunk in chat.send_message_stream(message):
            if chunk.text:
                yield chunk.text
                full_reply += chunk.text
            func_call = (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
                and getattr(chunk.candidates[0].content.parts[0], "function_call", None)
            )
            if func_call:
                collected_calls.append(func_call)
                involved_fc = True

        llm_end = time.time()
        llm_initial_duration = llm_end - llm_start

        # --- if we did call a tool, filter & run it ---
        if involved_fc:
            rag_start = time.time()
            parts, metadata, search_results = self._filter_fields_and_call_tool(collected_calls)
            rag_duration = time.time() - rag_start

            # 1) Log the full raw results for your cost/audit logs
            logger.debug(
                "Raw search_results for cost tracking: %s",
                json.dumps(search_results, ensure_ascii=False),
            )

            # 2) Yield frontend‐metadata if needed
            if metadata:
                yield metadata

            # 3) Stream the LLM’s follow-up (using only the minimal `parts`)
            llm_followup_start = time.time()
            for chunk in chat.send_message_stream(parts):
                if chunk.text:
                    yield chunk.text
                    full_reply += chunk.text

            llm_followup_end = time.time()
            llm_followup_duration = llm_followup_end - llm_followup_start

        # --- Persist conversation into Redis, but only minimal parts in history ---
        # Save the user’s message
        self.redis.save_message(user_id, Content(role="user", parts=[Part(text=message)]))

        if involved_fc:
            # Save only the function-response parts (fields_for_llm)
            self.redis.save_message(user_id, Content(role="model", parts=parts))

            # Then save the assistant’s natural-language reply
            assistant_msg = Content(role="model", parts=[Part(text=full_reply)])
        else:
            assistant_msg = Content(role="model", parts=[Part(text=full_reply)])

        self.redis.save_message(user_id, assistant_msg)

        total_end = time.time()
        total_duration = total_end - total_start

        prior_history = history + [Content(role="user", parts=[Part(text=message)])]
        prior_history = prior_history + [Content(role="model", parts=parts)] if involved_fc else prior_history
        token_in, token_out = self.num_tokens(prior_history, new_user_msg=message, assistant_reply=full_reply)

        troll_triggered = any(call.name == "trigger_troll_response" for call in collected_calls)
        logs = {
            "additional_info": "logs",
            "version": "1.0",
            "model": self.model_name,
            "user_id": user_id,
            "input_tokens": token_in,
            "output_tokens": token_out,
            "rag_speed": rag_duration if involved_fc else 0,
            "llm_speed": (llm_initial_duration + (llm_followup_duration if involved_fc else 0)),
            "troll_triggered": troll_triggered,
            "total_time": total_duration,
        }
        yield json.dumps(logs, ensure_ascii=False)


async def main_cli():
    import numpy as np

    from config.load_config import load_config

    prompts = load_config("config/prompts.yaml")
    sys_instruct = prompts.get("system_instructions")
    config = load_config("config/config.yaml")
    llm_cfg = config.get("llm", {})

    redis_store = RedisChatHistory()
    llm_client = LLMClient(
        model_name=llm_cfg.get("llm_model_name"),
        api_key=llm_cfg.get("GOOGLE_API_KEY"),
        sys_instruct=sys_instruct,
        config=config,
        redis_store=redis_store,
    )

    counter = np.random.randint(1, 99999)
    while True:
        user_msg = input("You: ")
        if user_msg.lower() in {"quit", "exit"}:
            break
        if user_msg.lower() == "reset":
            # reset not implemented – placeholder for future
            print("--- Chat session reset not implemented in refactor ---")
            continue

        print("LLM: ", end="", flush=True)
        async for chunk in llm_client.streaming_message(user_msg, user_id=counter):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_cli())
