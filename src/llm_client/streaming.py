from typing import AsyncGenerator, Callable, List

from google.genai.types import FinishReason, FunctionCall, Part

from constant import end_tag_info, end_tag_logs, start_tag_info, start_tag_logs


def has_reserved_tags(text: str) -> bool:
    """
    Return True if the text includes any frontend control tags that are not allowed
    to appear in LLM output.
    """
    return any(tag in text for tag in (start_tag_logs, end_tag_logs, start_tag_info, end_tag_info))


def strip_closing_question_tags() -> Callable[[AsyncGenerator[str, None]], AsyncGenerator[str, None]]:
    """
    Returns a function that wraps a generator and removes <closing_question> and </closing_question> tags
    with minimal buffering. Holds at most one chunk in memory to detect broken tags across chunk boundaries.
    """

    async def wrapper(stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        buffer = ""
        async for chunk in stream:
            combined = buffer + chunk

            # Check if any full tag appears in combined
            found_tag = False
            for tag in ("<closing_question>", "</closing_question>"):
                if tag in combined:
                    combined = combined.replace(tag, "")
                    found_tag = True

            if found_tag:
                # If tag was found and removed, drop buffer and yield combined result
                buffer = ""
                yield combined
            else:
                if buffer:
                    # No tag found, yield previous chunk
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

        if chunk.candidates[0].finish_reason != FinishReason.STOP and chunk.candidates[0].finish_reason in FinishReason:
            yield "gemini content filters triggered"


async def stream_llm_followup(
    session,
    parts: List[Part],
    remove_closing_question: bool = False,
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
            if remove_closing_question:
                converted = converted
            yield converted
