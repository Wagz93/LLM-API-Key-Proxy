"""
Anthropic SSE streaming conversion.

This module handles the conversion of OpenAI SSE streaming format to Anthropic's
streaming event format.

Anthropic's streaming format uses named events (message_start, content_block_delta, etc.)
while OpenAI uses a simpler data-only format.

This conversion is framework-agnostic and operates on async generators.
"""

import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional


def _generate_id(prefix: str = "msg") -> str:
    """Generate a unique ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class StreamingState:
    """
    Tracks state during streaming conversion.

    Anthropic's streaming format requires maintaining state across events
    to properly construct content_block events and track indices.
    """

    def __init__(self, original_model: str):
        self.original_model = original_model
        self.message_id = _generate_id("msg")
        self.content_block_index = 0
        self.current_content_type: Optional[str] = None
        self.current_tool_call_id: Optional[str] = None
        self.current_tool_name: Optional[str] = None
        self.tool_arguments_buffer: str = ""
        self.has_sent_message_start = False
        self.has_sent_content_block_start = False
        self.accumulated_text = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.stop_reason: Optional[str] = None
        # Track tool calls: OpenAI tool_index -> content block info
        self.tool_call_blocks: Dict[int, Dict[str, Any]] = {}
        # Map from OpenAI tool_index to assigned Anthropic content_block_index
        self.tool_index_to_content_index: Dict[int, int] = {}
        # Track the text content block index (if any)
        self.text_block_index: Optional[int] = None
        # Track if text block was closed
        self.text_block_closed: bool = False


def _format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format an SSE event in Anthropic's format."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _create_message_start_event(state: StreamingState) -> str:
    """Create the initial message_start event."""
    message = {
        "id": state.message_id,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": state.original_model,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": state.input_tokens, "output_tokens": 0},
    }
    return _format_sse_event("message_start", {"type": "message_start", "message": message})


def _create_content_block_start_event(
    index: int, block_type: str, extra: Optional[Dict[str, Any]] = None
) -> str:
    """Create a content_block_start event."""
    if block_type == "text":
        content_block = {"type": "text", "text": ""}
    elif block_type == "tool_use":
        content_block = {
            "type": "tool_use",
            "id": extra.get("id", _generate_id("toolu")) if extra else _generate_id("toolu"),
            "name": extra.get("name", "") if extra else "",
            "input": {},
        }
    else:
        content_block = {"type": block_type}

    return _format_sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": content_block},
    )


def _create_content_block_delta_event(
    index: int, block_type: str, content: Any
) -> str:
    """Create a content_block_delta event."""
    if block_type == "text":
        delta = {"type": "text_delta", "text": content}
    elif block_type == "tool_use":
        delta = {"type": "input_json_delta", "partial_json": content}
    else:
        delta = {"type": f"{block_type}_delta", "content": content}

    return _format_sse_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": delta},
    )


def _create_content_block_stop_event(index: int) -> str:
    """Create a content_block_stop event."""
    return _format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": index}
    )


def _create_message_delta_event(
    stop_reason: Optional[str], output_tokens: int
) -> str:
    """Create a message_delta event."""
    return _format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )


def _create_message_stop_event() -> str:
    """Create a message_stop event."""
    return _format_sse_event("message_stop", {"type": "message_stop"})


def _create_ping_event() -> str:
    """Create a ping event for keepalive."""
    return _format_sse_event("ping", {"type": "ping"})


def _convert_finish_reason(finish_reason: Optional[str], has_tool_calls: bool = False) -> Optional[str]:
    """Convert OpenAI finish_reason to Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"
    if finish_reason is None:
        return None

    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason, "end_turn")


async def convert_openai_stream_to_anthropic(
    openai_stream: AsyncGenerator[str, None],
    original_model: str,
    input_tokens: int = 0,
) -> AsyncGenerator[str, None]:
    """
    Convert an OpenAI SSE stream to Anthropic SSE format.

    Args:
        openai_stream: Async generator yielding OpenAI SSE chunks (as strings)
        original_model: The model name from the original Anthropic request
        input_tokens: Estimated input token count for the request

    Yields:
        Anthropic-formatted SSE event strings
    """
    state = StreamingState(original_model)
    state.input_tokens = input_tokens

    async for chunk in openai_stream:
        # Parse the OpenAI SSE chunk
        if not chunk.strip():
            continue

        # Handle the "data: " prefix
        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
        else:
            data_str = chunk.strip()

        # Skip [DONE] marker
        if data_str == "[DONE]":
            continue

        # Skip empty data
        if not data_str:
            continue

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Process the OpenAI chunk
        async for event in _process_openai_chunk(data, state):
            yield event

    # Finalize the stream
    async for event in _finalize_stream(state):
        yield event


async def _process_openai_chunk(
    data: Dict[str, Any], state: StreamingState
) -> AsyncGenerator[str, None]:
    """Process a single OpenAI chunk and yield Anthropic events."""
    # Send message_start if not yet sent
    if not state.has_sent_message_start:
        yield _create_message_start_event(state)
        state.has_sent_message_start = True

    # Extract the delta from the first choice
    choices = data.get("choices", [])
    if not choices:
        return

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    # Handle text content
    content = delta.get("content")
    if content:
        # Start text content block if needed
        if state.current_content_type != "text":
            # Close previous content block if any
            if state.has_sent_content_block_start:
                yield _create_content_block_stop_event(state.content_block_index)
                state.content_block_index += 1

            state.current_content_type = "text"
            state.text_block_index = state.content_block_index
            yield _create_content_block_start_event(state.content_block_index, "text")
            state.has_sent_content_block_start = True

        # Send content delta
        yield _create_content_block_delta_event(state.content_block_index, "text", content)
        state.accumulated_text += content

    # Handle tool calls
    tool_calls = delta.get("tool_calls", [])
    for tool_call in tool_calls:
        tool_index = tool_call.get("index", 0)
        tool_id = tool_call.get("id")
        function = tool_call.get("function", {})
        tool_name = function.get("name")
        arguments = function.get("arguments", "")

        # Check if this is a new tool call
        if tool_index not in state.tool_call_blocks:
            # Close previous text content block if any (before switching to tool_use)
            if state.has_sent_content_block_start and state.current_content_type == "text" and not state.text_block_closed:
                yield _create_content_block_stop_event(state.text_block_index)
                state.text_block_closed = True
                state.content_block_index += 1

            state.current_content_type = "tool_use"
            
            # Assign the next sequential content block index for this tool call
            assigned_index = state.content_block_index
            state.tool_index_to_content_index[tool_index] = assigned_index
            state.tool_call_blocks[tool_index] = {
                "id": tool_id or _generate_id("toolu"),
                "name": tool_name or "",
                "arguments": "",
                "content_index": assigned_index,
            }
            state.content_block_index += 1

            yield _create_content_block_start_event(
                assigned_index,
                "tool_use",
                {"id": state.tool_call_blocks[tool_index]["id"], "name": tool_name or ""},
            )
            state.has_sent_content_block_start = True

        # Update tool call info if provided
        if tool_name:
            state.tool_call_blocks[tool_index]["name"] = tool_name
        if tool_id:
            state.tool_call_blocks[tool_index]["id"] = tool_id

        # Send arguments delta
        if arguments:
            state.tool_call_blocks[tool_index]["arguments"] += arguments
            content_index = state.tool_index_to_content_index[tool_index]
            yield _create_content_block_delta_event(
                content_index, "tool_use", arguments
            )

    # Handle finish reason
    if finish_reason:
        has_tool_calls = bool(state.tool_call_blocks)
        state.stop_reason = _convert_finish_reason(finish_reason, has_tool_calls)

    # Handle usage information
    usage = data.get("usage")
    if usage:
        if "prompt_tokens" in usage:
            state.input_tokens = usage["prompt_tokens"]
        if "completion_tokens" in usage:
            state.output_tokens = usage["completion_tokens"]


async def _finalize_stream(state: StreamingState) -> AsyncGenerator[str, None]:
    """Finalize the stream by closing content blocks and sending final events."""
    # Close any open content blocks
    if state.has_sent_content_block_start:
        # Close text block if it exists and wasn't already closed
        if state.text_block_index is not None and not state.text_block_closed:
            yield _create_content_block_stop_event(state.text_block_index)
            state.text_block_closed = True

        # Close all tool call blocks using their assigned content indices
        for tool_index, block_info in state.tool_call_blocks.items():
            content_index = block_info["content_index"]
            yield _create_content_block_stop_event(content_index)

    # Send message_delta with stop_reason
    stop_reason = state.stop_reason or "end_turn"
    yield _create_message_delta_event(stop_reason, state.output_tokens)

    # Send message_stop
    yield _create_message_stop_event()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


async def create_error_event(error_type: str, error_message: str) -> str:
    """Create an Anthropic-format error event."""
    return _format_sse_event(
        "error",
        {
            "type": "error",
            "error": {"type": error_type, "message": error_message},
        },
    )
