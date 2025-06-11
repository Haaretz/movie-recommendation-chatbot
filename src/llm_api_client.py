import json
import time
from typing import AsyncGenerator, List, Optional

from google import genai
from google.genai import types
from google.genai.types import Content, FunctionCall, Part

from config.models import (
    ChatConfig,
    ChatContext,
    EmbeddingConfig,
    FieldsConfig,
    LLMConfig,
    QdrantConfig,
)
from constant import (
    NO_RESULT,
    TROLL,
    end_tag_info,
    end_tag_logs,
    start_tag_info,
    start_tag_logs,
)
from logger import logger
from src.redis_chat_history import RedisChatHistory
from src.tools.import_tools import qdrant_tools
from src.tools.search.search_article_core import SearchArticle


class LLMClient:
    """High‑level wrapper around Gemini chat + article search with streaming I/O."""

    def __init__(
        self,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        qdrant_config: QdrantConfig,
        fields_config: FieldsConfig,
        sys_instruct: str,
        redis_store: RedisChatHistory,
        chat_config: ChatConfig,
    ):
        self.search_article = SearchArticle(qdrant_config, embedding_config, chat_config)
        self.sys_instruct = sys_instruct
        self.model_name = llm_config.llm_model_name
        self.api_key = llm_config.GOOGLE_API_KEY
        self.fields_for_frontend = fields_config.fields_for_frontend
        self.fields_for_llm = fields_config.fields_for_llm
        self.redis = redis_store
        self.chat_config = chat_config
        self.user_quota = chat_config.max_user_messages_per_session

        self.client = genai.Client(vertexai=False, api_key=self.api_key)

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
        if not resp.text:
            return query
        translated = resp.text.strip()
        logger.debug("Translated query to Hebrew: '%s'", translated)
        return translated

    def _filter_fields_and_call_tool(self, function_calls, ctx: ChatContext):
        """
        Process function calls, route to appropriate handlers, and return
        the response parts, frontend metadata, and raw search results.
        """
        parts = []
        search_results = None

        for call in function_calls:
            if call.name == "get_dataset_articles":
                handler_parts, search_results = self._handle_get_dataset_articles(call, ctx)
                parts.extend(handler_parts)
            elif call.name == "trigger_troll_response":
                handler_parts, search_results = self._handle_trigger_troll_response(call)
                parts.extend(handler_parts)
            else:
                logger.warning(f"No handler registered for function '{call.name}'")

        # Prepare metadata for frontend if we have list results
        metadata = None
        if isinstance(search_results, list):
            metadata = [{k: item.get(k, None) for k in self.fields_for_frontend} for item in search_results]
        return parts, metadata

    def _handle_get_dataset_articles(self, call, ctx: ChatContext):
        """
        Handle 'get_dataset_articles' calls by performing the search and
        formatting the response part.
        """
        args = call.args
        query = args.get("query")
        streaming = args.get("streaming_platforms", None)
        genres = args.get("genres", None)
        media_type = args.get("media_type", None)
        writer_filter = args.get("writer_filter", None)

        if not query:
            logger.warning("No query provided for get_dataset_articles")
            query = ctx.message

        logger.info(
            "Executing get_dataset_articles with query=%s, streaming=%s, genres=%s, media_type=%s, writer_filter=%s",
            query,
            streaming,
            genres,
            media_type,
            writer_filter,
        )

        translated_query = self._translate_english_query(query)
        search_results = self.search_article.retrieve_relevant_documents(
            translated_query, streaming, genres, media_type, writer_filter, ctx.seen
        )

        if not search_results:
            # No results found, return a placeholder response
            part = Part.from_function_response(
                name=call.name,
                response={"content": [NO_RESULT]},
            )
            return [part], None

        # Convert results for LLM
        content_list = [{k: item.get(k) for k in self.fields_for_llm} for item in search_results]
        # for d in content_list:
        #     d.pop('article_id', None) # remove article_id from the response. unnecessary for LLM
        part = Part.from_function_response(name=call.name, response={"content": content_list})
        return [part], search_results

    def _handle_trigger_troll_response(self, call):
        """
        Handle 'trigger_troll_response' calls by returning predefined troll content.
        """
        logger.debug("Triggering troll response")

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

    @staticmethod
    def contains_disallowed_tags(text: str) -> bool:
        return any(sub in text for sub in [start_tag_logs, end_tag_logs, start_tag_info, end_tag_info])

    def _get_message_quota(self, user_id: str) -> tuple[int, str | None]:
        """
        Consume one message credit for the user.
        Returns:
          - blocked (bool): True if user is out of credits.
          - warning (str|None): A warning or blocked message to show.
          - is_last_allowed (bool): True if this is the user's final allowed message.
        """
        used = self.redis.get_usage_count(user_id)
        remaining = self.user_quota - used

        if remaining <= 0:
            return remaining, self.chat_config.blocked_message

        self.redis.increment_usage_counter(user_id)

        if remaining == 1:
            return remaining, self.chat_config.warn_last_message

        if remaining <= 3:
            return remaining, self.chat_config.warn_template.format(remaining=remaining - 1)

        return remaining, None

    @staticmethod
    def _extract_seen_article_ids(history: List[Content]) -> set[str]:
        """
        Given a Redis-loaded chat history, pull out all the article IDs
        returned by previous get_dataset_articles calls.
        """
        seen_ids: set[str] = set()
        for content in history:
            for part in content.parts:
                # only function_response parts have .function_response
                fc = getattr(part, "function_response", None)
                if fc and fc.name == "get_dataset_articles":
                    # assume response is {"content": [{... "id": ...}, ...]}
                    for item in fc.response.get("content", []):
                        if isinstance(item, dict):
                            article_id = item.get("article_id")
                            if article_id:
                                seen_ids.add(article_id)
        return seen_ids

    async def regenerate_response(self, sso_id: str, session_id: str) -> AsyncGenerator[str, None]:
        """Regenerate the response for the last user message, reusing streaming logic."""
        # TODO: consider deferring this deletion after response yield for lower perceived latency
        conversation_key = f"{sso_id}_{session_id}"
        blocked, warning = self._get_message_quota(conversation_key)
        if blocked:
            yield warning
            return

        message = self.redis.pop_last_conversation(conversation_key)

        # Inject warning into the message itself
        full_message = f"{warning}\n{message}" if warning else message

        history = self.redis.load_history(conversation_key)
        seen = self._extract_seen_article_ids(history)

        ctx = ChatContext(
            conversation_key=conversation_key,
            sso_id=sso_id,
            session_id=session_id,
            message=full_message,
            history=history,
            seen=seen,
            remaining_user_messages=blocked,
        )

        async for chunk in self._process_message_stream(ctx, regenerate=True):
            yield chunk

    async def streaming_message(
        self, message: str, session_id: str, sso_id: str, _error_count: int
    ) -> AsyncGenerator[str, None]:
        """Handle a new user message with full persistence."""
        conversation_key = f"{sso_id}_{session_id}"

        remaining, warning = self._get_message_quota(conversation_key)
        if remaining == 0:
            yield warning
            return

        # Prepend warning if needed
        full_message = f"{warning}\n{message}" if warning else message

        history = self.redis.load_history(conversation_key)
        seen = self._extract_seen_article_ids(history)

        ctx = ChatContext(
            conversation_key=conversation_key,
            sso_id=sso_id,
            session_id=session_id,
            message=full_message,
            history=history,
            seen=seen,
            remaining_user_messages=remaining,
            error_count=_error_count,
        )

        async for chunk in self._process_message_stream(ctx):
            yield chunk

    @staticmethod
    def _wrap_info(teaser: dict, last_message: bool = False) -> str:
        if last_message:
            payload = {"teasers": teaser, "system": {"last_message": True}}
        else:
            payload = {"teasers": teaser, "system": {"last_message": False}}
        return f"{start_tag_info}{json.dumps(payload, ensure_ascii=False)}{end_tag_info}"

    @staticmethod
    def _convert_streaming_markdown_bold(text: str, bold_open: bool) -> tuple[str, bool]:
        """
        Scan `text` for '**', toggling between opening and closing <strong> tags.
        Returns the converted text and updated bold_open state.
        """
        parts = text.split("**")
        # if there are no '**', nothing changes
        if len(parts) == 1:
            return text, bold_open

        out = []
        for i, segment in enumerate(parts):
            out.append(segment)
            # after every segment except the last, inject a tag
            if i < len(parts) - 1:
                if bold_open:
                    out.append("</strong>")
                else:
                    out.append("<strong>")
                bold_open = not bold_open

        return "".join(out), bold_open

    async def _process_message_stream(self, ctx: ChatContext, regenerate: bool = False) -> AsyncGenerator[str, None]:
        chat = self._create_chat_session(history=ctx.history)
        full_reply = ""
        collected_calls: List[FunctionCall] = []
        involved_fc = False

        total_start = time.time()
        llm_initial_duration = 0
        rag_duration = 0
        llm_followup_duration = 0

        # --- Step 1: Stream initial LLM response ---
        llm_start = time.time()
        async for chunk in self._stream_llm_response(chat, ctx.message, collected_calls):
            if chunk == "DISALLOWED_TAGS":
                yield "Error: Disallowed tags detected in the response."
                return
            full_reply += chunk
            yield chunk
        llm_initial_duration = time.time() - llm_start

        if len(collected_calls) == 0 and ctx.remaining_user_messages == 1:
            yield self._wrap_info({}, last_message=True)

        # --- Step 2: Handle Function Calls ---
        parts = None
        metadata = None

        if collected_calls:
            involved_fc = True
            rag_start = time.time()
            parts, metadata = self._filter_fields_and_call_tool(collected_calls, ctx)
            rag_duration = time.time() - rag_start

            if metadata and ctx.remaining_user_messages == 1:
                yield self._wrap_info(metadata, last_message=True)
            elif metadata:
                yield self._wrap_info(metadata, last_message=False)

            llm_followup_start = time.time()
            async for chunk in self._stream_llm_followup(chat, parts):
                if chunk == "DISALLOWED_TAGS":
                    yield "Error: Disallowed tags detected in the response."
                    return
                full_reply += chunk
                yield chunk
            llm_followup_duration = time.time() - llm_followup_start

        # --- Step 3: Save to Redis and log ---
        self._save_chat(ctx.message, involved_fc, full_reply, ctx.conversation_key, parts if involved_fc else None)
        total_duration = time.time() - total_start

        durations = {
            "llm_initial": llm_initial_duration,
            "rag": rag_duration,
            "llm_followup": llm_followup_duration,
            "total": total_duration,
            "remaining_user_messages": ctx.remaining_user_messages,
            "thinking process": "Thinking Process:" in full_reply,
        }

        yield self._generate_logs(
            ctx=ctx,
            model=self.model_name,
            collected_calls=collected_calls,
            full_reply=full_reply,
            involved_fc=involved_fc,
            parts=parts if involved_fc else None,
            durations=durations,
            regenerate=regenerate,
            article_ids=[item.get("article_id") for item in metadata] if metadata else None,
        )

    async def _stream_llm_response(
        self, chat, message: str, collected_calls: List[FunctionCall]
    ) -> AsyncGenerator[str, None]:
        bold_open = False
        # _in_thinking_process = False

        for chunk in chat.send_message_stream(message):
            if chunk.text:
                if self.contains_disallowed_tags(chunk.text):
                    yield "DISALLOWED_TAGS"
                    return

                # Detect start of Thinking Process block
                # if 'Thinking Process' in chunk.text:
                #     _in_thinking_process = True
                #     logger.warning("Thinking Process started")

                # # If inside Thinking Process, always yield until Hebrew after any newline
                # if _in_thinking_process:
                #     # End Thinking Process when a newline is followed by Hebrew text anywhere in this chunk
                #     if re.search(r"\n[\u0590-\u05FF\uFB1D-\uFB4F]", chunk.text):
                #         _in_thinking_process = False
                #     yield chunk.text
                #     continue

                converted, bold_open = self._convert_streaming_markdown_bold(chunk.text, bold_open)
                yield converted

            func_call = (
                chunk.candidates
                and chunk.candidates[0].content
                and chunk.candidates[0].content.parts
                and getattr(chunk.candidates[0].content.parts[0], "function_call", None)
            )
            if func_call:
                collected_calls.append(func_call)

    async def _stream_llm_followup(self, chat, parts: List[Part]) -> AsyncGenerator[str, None]:
        # _in_thinking_process = False
        bold_open = False

        for chunk in chat.send_message_stream(parts):
            if chunk.text:
                text = chunk.text

                # Detect start of Thinking Process block
                # if text.startswith("Thinking Process"):
                #     self._in_thinking_process = True
                #     logger.warning("Thinking Process started")

                # # If inside Thinking Process, always yield until Hebrew after any newline
                # if _in_thinking_process:
                #     # End Thinking Process when a newline is followed by Hebrew text anywhere in this chunk
                #     if re.search(r"\n[\u0590-\u05FF\uFB1D-\uFB4F]", text):
                #         _in_thinking_process = False
                #     yield text
                #     continue

                # Normal disallowed tags handling
                if self.contains_disallowed_tags(text):
                    yield "DISALLOWED_TAGS"
                    return

                converted, bold_open = self._convert_streaming_markdown_bold(chunk.text, bold_open)
                yield converted

    def _generate_logs(
        self,
        ctx: ChatContext,
        model: str,
        collected_calls: List[FunctionCall],
        full_reply: str,
        involved_fc: bool,
        parts: Optional[List[Part]],
        durations: dict,
        regenerate: bool,
        ids: Optional[List[str]] = None,
    ) -> str:
        prior_history = ctx.history + [Content(role="user", parts=[Part(text=ctx.message)])]
        if involved_fc and parts:
            prior_history += [Content(role="model", parts=parts)]

        token_in, token_out = self.num_tokens(prior_history, new_user_msg=ctx.message, assistant_reply=full_reply)

        troll_triggered = any(call.name == "trigger_troll_response" for call in collected_calls)
        logs = {
            "additional_info": {
                "version": "1.0",
                "conversation_key": ctx.conversation_key,
                "sso_id": ctx.sso_id,
                "session_id": ctx.session_id,
                "model": model,
                "input_tokens": token_in,
                "output_tokens": token_out,
                "rag_speed": durations.get("rag", 0),
                "llm_speed": durations.get("llm_initial", 0) + durations.get("llm_followup", 0),
                "function_calls_args": [
                    {k: call.args.get(k) for k in call.args} for call in collected_calls if call.args
                ],
                "troll_triggered": troll_triggered,
                "total_time": durations.get("total", 0),
                "regenerate": regenerate,
                "remaining_user_messages": durations.get("remaining_user_messages", 0),
                "timestamp": time.time(),
                "article_ids": ids if ids else [],
                "thinking_process": durations.get("thinking process", False),
            }
        }
        return start_tag_logs + json.dumps(logs, ensure_ascii=False) + end_tag_logs

    def _save_chat(
        self,
        message: str,
        involved_fc: bool,
        full_reply: str,
        user_id: str,
        parts: Optional[List[Part]] = None,
    ):
        """
        Save the user message and assistant reply to Redis.
        If involved_fc is True, save only the function-response parts.
        """
        self.redis.save_message(user_id, Content(role="user", parts=[Part(text=message)]))

        if involved_fc:
            # Save only the function-response parts (fields_for_llm)
            self.redis.save_message(user_id, Content(role="model", parts=parts))

            # Then save the assistant’s natural-language reply
            assistant_msg = Content(role="model", parts=[Part(text=full_reply)])
        else:
            assistant_msg = Content(role="model", parts=[Part(text=full_reply)])

        self.redis.save_message(user_id, assistant_msg)


async def main_cli():
    import numpy as np

    from config.loader import load_config

    app_config = load_config()

    # Load system instructions from prompts.yaml
    import yaml

    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    sys_instruct = prompts.get("system_instructions", "")

    llm_client = LLMClient(
        llm_config=app_config.llm,
        embedding_config=app_config.embedding,
        qdrant_config=app_config.qdrant,
        fields_config=app_config.fields,
        sys_instruct=sys_instruct,
        redis_store=RedisChatHistory(app_config.chat.chat_ttl_seconds),
        chat_config=app_config.chat,
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
        async for chunk in llm_client.streaming_message(
            user_msg, session_id=str(counter), sso_id=str(counter), _error_count=0
        ):
            print(chunk, end="", flush=True)
        print()

        # async for chunk in llm_client.regenerate_response(user_id=counter):
        #     print(chunk, end="", flush=True)

        print()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_cli())
