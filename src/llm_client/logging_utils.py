import json
import time
from typing import List

from google.genai.types import FunctionCall

from config.models import ChatContext


def extract_function_args(calls: List[FunctionCall]) -> List[dict]:
    return [{k: call.args.get(k) for k in call.args} for call in calls if call.args]


def compute_llm_speed(durations: dict) -> float:
    return durations.get("llm_initial", 0.0) + durations.get("llm_followup", 0.0)


def was_troll_triggered(calls: List[FunctionCall]) -> bool:
    return any(call.name == "trigger_troll_response" for call in calls)


def generate_log_blob(
    ctx: ChatContext,
    collected_calls: List[FunctionCall],
    token_in: int,
    token_out: int,
    durations: dict,
    regenerate: bool,
) -> str:
    """
    Return a structured JSON blob with full metadata.
    Intended for streaming inside <log>...</log> tags.
    """

    log_payload = {
        "version": "1.0",
        "conversation_key": ctx.conversation_key,
        "sso_id": ctx.sso_id,
        "session_id": ctx.session_id,
        "model": getattr(ctx, "model", "unknown"),
        "input_tokens": token_in,
        "output_tokens": token_out,
        "rag_speed": durations.get("rag", 0),
        "llm_speed": compute_llm_speed(durations),
        "function_calls_args": extract_function_args(collected_calls),
        "troll_triggered": was_troll_triggered(collected_calls),
        "total_time": durations.get("total", 0),
        "regenerate": regenerate,
        "remaining_user_messages": durations.get("remaining_user_messages", 0),
        "timestamp": time.time(),
        "article_ids": durations.get("article_ids", []),
        "thinking_process": durations.get("thinking process", False),
    }

    return json.dumps({"additional_info": log_payload}, ensure_ascii=False)
