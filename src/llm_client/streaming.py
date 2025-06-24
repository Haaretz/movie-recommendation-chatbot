from typing import AsyncGenerator, Callable, List

from google.genai.types import FinishReason, FunctionCall, Part

from constant import (
    BOLD_HTML_PATTERN,
    end_tag_info,
    end_tag_logs,
    start_tag_info,
    start_tag_logs,
)


def has_reserved_tags(text: str) -> bool:
    """
    Return True if the text includes any frontend control tags that are not allowed
    to appear in LLM output.
    """
    return any(tag in text for tag in (start_tag_logs, end_tag_logs, start_tag_info, end_tag_info))


def strip_closing_question_tags(
    strip_tags: bool = True,
) -> Callable[[AsyncGenerator[str, None]], AsyncGenerator[str, None]]:
    """
    Returns a function that wraps a generator and either removes or preserves
    <closing_question> and </closing_question> tags, depending on `strip_tags`.
    Ensures tags are never split across chunks by buffering one chunk.
    """

    async def wrapper(stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        buffer = ""
        async for chunk in stream:
            combined = buffer + chunk

            # Always check for complete tags in the combined string
            has_full_tag = any(tag in combined for tag in ("<closing_question>", "</closing_question>"))
            has_full_bold = bool(BOLD_HTML_PATTERN.search(combined))

            if has_full_tag or has_full_bold:
                if strip_tags:
                    # Remove both tags in a single line
                    combined = combined.replace("<closing_question>", "").replace("</closing_question>", "")
                buffer = ""

                def _clean(match):
                    return match.group(0).replace(r"\"", '"')  # Replace escaped quotes with actual quotes

                combined = BOLD_HTML_PATTERN.sub(_clean, combined)
                yield combined
            else:
                if buffer:
                    yield buffer
                buffer = chunk

        # After stream ends, flush remaining
        if buffer:
            yield buffer

    return wrapper


def convert_streaming_markdown_bold(text: str, bold_open: bool) -> tuple[str, bool]:
    """
    Convert streaming markdown '**bold**' to HTML <strong> tags.
    Because streaming may split text mid-bold, we keep track of bold state.
    """
    parts = text.split("**")
    if len(parts) == 1:
        return text, bold_open

    out = []
    for i, segment in enumerate(parts):
        out.append(segment)
        if i < len(parts) - 1:
            out.append("</strong>" if bold_open else "<strong>")
            bold_open = not bold_open
    return "".join(out), bold_open


async def stream_llm_response(
    session,
    user_message: str,
    collected_calls: List[FunctionCall],
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM response to a user message.
    Collects any function calls triggered along the way.
    """
    bold_open = False

    for chunk in session.stream(user_message):
        if chunk.text:
            if has_reserved_tags(chunk.text):
                yield "DISALLOWED_TAGS"
                return
            converted, bold_open = convert_streaming_markdown_bold(chunk.text, bold_open)
            yield converted

        func_call = (
            chunk.candidates
            and chunk.candidates[0].content
            and chunk.candidates[0].content.parts
            and getattr(chunk.candidates[0].content.parts[0], "function_call", None)
        )
        if func_call:
            collected_calls.append(func_call)

        reason = chunk.candidates[0].finish_reason
        if reason is not None and reason != FinishReason.STOP and reason in FinishReason:
            yield "gemini content filters triggered"


async def stream_llm_followup(
    session,
    parts: List[Part],
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM continuation after a function call response (RAG).
    """
    bold_open = False

    for chunk in session.stream(parts):
        if chunk.text:
            if has_reserved_tags(chunk.text):
                yield "DISALLOWED_TAGS"
                return
            converted, bold_open = convert_streaming_markdown_bold(chunk.text, bold_open)
            yield converted
