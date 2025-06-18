from typing import List, Optional, Tuple

from google.genai.types import FunctionCall, Part

from config.models import ChatContext
from constant import NO_RESULT, TROLL
from logger import logger


class DatasetArticleHandler:
    def __init__(self, search_article, fields_for_llm, fields_for_frontend, translate_func):
        self.search_article = search_article
        self.fields_for_llm = fields_for_llm
        self.fields_for_frontend = fields_for_frontend
        self.translate = translate_func

    def __call__(self, call: FunctionCall, ctx: ChatContext) -> Tuple[List[Part], Optional[list]]:
        query, filters = self._extract_args(call.args, fallback_query=ctx.message)

        logger.info(
            "Executing get_dataset_articles with query=%s, streaming=%s, genres=%s, media_type=%s, writer_filter=%s",
            query,
            filters["streaming"],
            filters["genres"],
            filters["media_type"],
            filters["writer_filter"],
        )

        translated_query = self.translate(query)
        results = self._run_search(translated_query, filters, ctx.seen)

        if not results:
            return [self._wrap_no_result(call.name)], None

        return [self._wrap_results(call.name, results)], results

    def _extract_args(self, args: dict, fallback_query: str) -> Tuple[str, dict]:
        return (
            args.get("query") or fallback_query,
            {
                "streaming": args.get("streaming_platforms"),
                "genres": args.get("genres"),
                "media_type": args.get("media_type"),
                "writer_filter": args.get("writer_filter"),
            },
        )

    def _run_search(self, query: str, filters: dict, seen_ids: set) -> list:
        return self.search_article.retrieve_relevant_documents(
            query,
            filters["streaming"],
            filters["genres"],
            filters["media_type"],
            filters["writer_filter"],
            seen_ids,
        )

    def _wrap_no_result(self, name: str) -> Part:
        return Part.from_function_response(name=name, response={"content": [NO_RESULT]})

    def _wrap_results(self, name: str, results: list) -> Part:
        content_list = [{k: item.get(k) for k in self.fields_for_llm} for item in results]
        return Part.from_function_response(name=name, response={"content": content_list})


class TriggerTrollResponseHandler:
    def __init__(self, fields_for_llm):
        self.fields_for_llm = fields_for_llm

    def __call__(self, call: FunctionCall, ctx: ChatContext) -> Tuple[List[Part], list]:
        logger.debug("Triggering troll response")
        return [self._wrap_troll_response(call.name)], [TROLL]

    def _wrap_troll_response(self, name: str) -> Part:
        content = [{k: item.get(k) for k in self.fields_for_llm} for item in [TROLL]]
        return Part.from_function_response(name=name, response={"content": content})


def build_handler_registry(
    search_article,
    fields_for_llm,
    fields_for_frontend,
    translate_func,
):
    return {
        "get_dataset_articles": DatasetArticleHandler(
            search_article=search_article,
            fields_for_llm=fields_for_llm,
            fields_for_frontend=fields_for_frontend,
            translate_func=translate_func,
        ),
        "trigger_troll_response": TriggerTrollResponseHandler(fields_for_llm=fields_for_llm),
    }
