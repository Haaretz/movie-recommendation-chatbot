import json
import time
from typing import AsyncGenerator, Optional, Set

from google import genai
from google.genai.types import Content, FunctionCall, Part

from config.models import (
    ChatConfig,
    ChatContext,
    EmbeddingConfig,
    FieldsConfig,
    LLMConfig,
    QdrantConfig,
)
from constant import end_tag_info, end_tag_logs, start_tag_info, start_tag_logs
from src.llm_client.handlers import build_handler_registry
from src.llm_client.logging_utils import generate_log_blob
from src.llm_client.session import LLMChatSession
from src.llm_client.streaming import (
    stream_llm_followup,
    stream_llm_response,
    strip_closing_question_tags,
)
from src.llm_client.tokenizer import Tokenizer
from src.redis_chat_history import RedisChatHistory
from src.tools.import_tools import qdrant_tools
from src.tools.search.search_article_core import SearchArticle


class LLMClient:
    def __init__(
        self,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        qdrant_config: QdrantConfig,
        fields_config: FieldsConfig,
        sys_instruct: str,
        redis_store: RedisChatHistory,
        chat_config: ChatConfig,
        excluded_ids: Optional[Set[str]] = None,
    ):
        self.sys_instruct = sys_instruct
        self.model_name = llm_config.llm_model_name
        self.api_key = llm_config.GOOGLE_API_KEY
        self.redis = redis_store
        self.chat_config = chat_config
        self.user_quota = chat_config.max_user_messages_per_session
        self.fields_for_frontend = fields_config.fields_for_frontend
        self.fields_for_llm = fields_config.fields_for_llm
        excluded_ids = excluded_ids if excluded_ids is not None else set()

        self.search_article = SearchArticle(qdrant_config, embedding_config, chat_config, excluded_ids=excluded_ids)

        self.handlers = build_handler_registry(
            search_article=self.search_article,
            fields_for_llm=self.fields_for_llm,
            fields_for_frontend=self.fields_for_frontend,
            translate_func=self._translate_english_query,
        )

        self.tokenizer = Tokenizer(self.model_name)
        self.client = genai.Client(vertexai=False, api_key=self.api_key)

    def _translate_english_query(self, query: str) -> str:
        is_hebrew = any("\u0590" <= ch <= "\u05FF" or "\uFB1D" <= ch <= "\uFB4F" for ch in query)
        if is_hebrew:
            return query

        prompt = f"Translate the following English query to Hebrew: '{query}'. Return only the translation."
        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return resp.text.strip() if resp.text else query

    def _wrap_info(self, teaser: dict, last_message: bool) -> str:
        payload = {"teasers": teaser, "system": {"last_message": last_message}}
        return f"{start_tag_info}{json.dumps(payload, ensure_ascii=False)}{end_tag_info}"

    def _consume_quota(self, user_id: str) -> tuple[int, Optional[str]]:
        used = self.redis.get_usage_count(user_id)
        remaining_before = self.user_quota - used
        print(f"--------- User messages remaining (used: {used}) ---------")

        if remaining_before <= 0:
            return -1, self.chat_config.blocked_message

        self.redis.increment_usage_counter(user_id)
        remaining_after = remaining_before - 1

        if remaining_after == 0:
            return 0, self.chat_config.warn_last_message

        elif remaining_after == 1:
            warning_text = "You can send one final message."
        elif remaining_after <= 2:
            warning_text = f"You can send {remaining_after} more messages."
        else:
            warning_text = None
        if warning_text:
            return remaining_after, self.chat_config.warn_template.format(warning_text=warning_text)
        else:
            return remaining_after, None

    def _extract_seen_article_ids(self, history: list[Content]) -> set[str]:
        seen_ids = set()
        for content in history:
            for part in content.parts:
                fc = getattr(part, "function_response", None)
                if fc and fc.name == "get_dataset_articles":
                    for item in fc.response.get("content", []):
                        if isinstance(item, dict) and item.get("article_id"):
                            seen_ids.add(item["article_id"])
        return seen_ids

    async def stream_chat(
        self,
        message: str,
        session_id: str,
        sso_id: str,
        _error_count: int,
        regenerate: bool = False,
    ) -> AsyncGenerator[str, None]:
        key = f"{sso_id}_{session_id}"
        remaining, warning = self._consume_quota(key)
        if remaining == -1:
            yield warning
            return

        full_msg = f"{warning}\nUser query: {message}" if warning else message
        history = self.redis.load_history(key)
        seen = self._extract_seen_article_ids(history)

        ctx = ChatContext(
            conversation_key=key,
            sso_id=sso_id,
            session_id=session_id,
            message=full_msg,
            history=history,
            seen=seen,
            remaining_user_messages=remaining,
            error_count=_error_count,
        )

        async for chunk in self._run_full_chat(ctx, regenerate=regenerate):
            yield chunk

    async def regenerate_response(
        self,
        session_id: str,
        sso_id: str,
        _error_count: int,
    ) -> AsyncGenerator[str, None]:
        key = f"{sso_id}_{session_id}"
        remaining, warning = self._consume_quota(key)
        if remaining == -1:
            yield warning
            return

        message = self.redis.pop_last_conversation(key)
        full_msg = f"{warning}\n{message}" if warning else message
        history = self.redis.load_history(key)
        seen = self._extract_seen_article_ids(history)

        ctx = ChatContext(
            conversation_key=key,
            sso_id=sso_id,
            session_id=session_id,
            message=full_msg,
            history=history,
            seen=seen,
            remaining_user_messages=remaining,
            error_count=_error_count,
        )

        async for chunk in self._run_full_chat(ctx, regenerate=True):
            yield chunk

    async def _run_full_chat(self, ctx: ChatContext, regenerate: bool = False) -> AsyncGenerator[str, None]:
        chat = LLMChatSession(self.client, self.model_name, self.sys_instruct, ctx.history, tools=[qdrant_tools])

        collected_calls: list[FunctionCall] = []
        full_reply = ""
        durations = {"thinking process": "Thinking Process:" in ctx.message}
        remove_closing_question = False
        start_total = time.time()
        start_llm = time.time()

        stripper = strip_closing_question_tags()
        raw_stream = stream_llm_response(chat, ctx.message, collected_calls)

        async for chunk in stripper(raw_stream):
            full_reply += chunk
            yield chunk
        durations["llm_initial"] = time.time() - start_llm

        if len(collected_calls) == 0 and ctx.remaining_user_messages == 1:
            yield self._wrap_info({}, last_message=True)

        metadata, parts = None, []
        if collected_calls:
            start_rag = time.time()
            for call in collected_calls:
                handler = self.handlers.get(call.name)
                if not handler:
                    continue
                handler_parts, results = handler(call, ctx)
                parts.extend(handler_parts)
                if isinstance(results, list):
                    metadata = [{k: r.get(k, None) for k in self.fields_for_frontend} for r in results]
                if call.name == "get_dataset_articles" and not results:
                    remove_closing_question = True
            durations["rag"] = time.time() - start_rag

            if metadata:
                yield self._wrap_info(metadata, last_message=(ctx.remaining_user_messages == 1))

            start_followup = time.time()
            async for chunk in stream_llm_followup(chat, parts, remove_closing_question=remove_closing_question):
                full_reply += chunk
                yield chunk
            durations["llm_followup"] = time.time() - start_followup

        self._save_to_redis(ctx.message, full_reply, ctx.conversation_key, parts if parts else None)
        durations["total"] = time.time() - start_total
        durations["remaining_user_messages"] = ctx.remaining_user_messages

        token_in, token_out = self.tokenizer.count_tokens(ctx.history, ctx.message, full_reply)

        yield f"{start_tag_logs}{generate_log_blob(ctx, collected_calls, token_in, token_out, durations, regenerate)}{end_tag_logs}"

    def _save_to_redis(self, msg: str, reply: str, key: str, parts: Optional[list[Part]]):
        self.redis.save_message(key, Content(role="user", parts=[Part(text=msg)]))
        if parts:
            self.redis.save_message(key, Content(role="model", parts=parts))
        self.redis.save_message(key, Content(role="model", parts=[Part(text=reply)]))


if __name__ == "__main__":
    import asyncio

    import numpy as np
    import yaml

    from config.excluded_config import ExcludedIdLoader
    from config.loader import load_config
    from src.redis_chat_history import RedisChatHistory

    async def main_cli():
        app_config = load_config()

        # Load excluded IDs
        excluded_ids = ExcludedIdLoader().get_excluded_ids()

        # Load system instructions from YAML
        with open("config/prompts.yaml", "r") as f:
            prompts = yaml.safe_load(f)
        sys_instruct = prompts.get("system_instructions", "")

        client = LLMClient(
            llm_config=app_config.llm,
            embedding_config=app_config.embedding,
            qdrant_config=app_config.qdrant,
            fields_config=app_config.fields,
            sys_instruct=sys_instruct,
            redis_store=RedisChatHistory(app_config.chat.chat_ttl_seconds),
            chat_config=app_config.chat,
            excluded_ids=excluded_ids,
        )

        session_id = str(np.random.randint(100000))
        sso_id = session_id

        while True:
            user_msg = input("You: ").strip()
            if user_msg.lower() in {"quit", "exit"}:
                break
            if user_msg.lower() == "reset":
                print("[reset not implemented in CLI]")
                continue

            print("LLM: ", end="", flush=True)
            async for chunk in client.stream_chat(
                message=user_msg,
                session_id=session_id,
                sso_id=sso_id,
                _error_count=0,
            ):
                print(chunk, end="", flush=True)
            print()
            print("\n--- Regenerating response ---")
            async for chunk in client.regenerate_response(session_id=session_id, sso_id=session_id, _error_count=0):
                print(chunk, end="", flush=True)
            print()

    asyncio.run(main_cli())
