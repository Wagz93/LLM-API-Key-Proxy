"""
Anthropic ↔ OpenAI format translator.

This module provides bidirectional translation between Anthropic's Messages API format
and OpenAI's Chat Completions API format.

The translation layer is framework-agnostic and operates on dictionaries,
making it usable in any Python context.

Translation Flow:
1. Anthropic Request → OpenAI Request (request_to_openai)
2. OpenAI Response → Anthropic Response (response_from_openai)
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union


def _generate_id(prefix: str = "msg") -> str:
    """Generate a unique ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# =============================================================================
# REQUEST TRANSLATION: Anthropic → OpenAI
# =============================================================================


def _convert_anthropic_content_to_openai(
    content: Union[str, List[Dict[str, Any]]]
) -> Union[str, List[Dict[str, Any]]]:
    """
    Convert Anthropic content blocks to OpenAI content format.

    Anthropic uses typed content blocks (text, image, tool_result, etc.)
    OpenAI uses a similar but slightly different structure.
    """
    if isinstance(content, str):
        return content

    openai_content: List[Dict[str, Any]] = []

    for block in content:
        block_type = block.get("type", "text")

        if block_type == "text":
            openai_content.append({"type": "text", "text": block.get("text", "")})

        elif block_type == "image":
            # Convert Anthropic image format to OpenAI format
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
            elif source.get("type") == "url":
                openai_content.append(
                    {"type": "image_url", "image_url": {"url": source.get("data", "")}}
                )

        elif block_type == "tool_result":
            # Tool results are handled specially in message conversion
            # Just pass through for now
            openai_content.append(block)

        elif block_type == "tool_use":
            # Tool use blocks in content are not common, but handle them
            openai_content.append(block)

        else:
            # Unknown block type, try to preserve as text if possible
            if "text" in block:
                openai_content.append({"type": "text", "text": block.get("text", "")})

    return openai_content if openai_content else ""


def _convert_anthropic_message_to_openai(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single Anthropic message to OpenAI message(s).

    This may produce multiple OpenAI messages in some cases,
    such as when tool results need to be converted to tool messages.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")

    # Handle tool results specially
    if isinstance(content, list):
        tool_result_blocks = [b for b in content if b.get("type") == "tool_result"]
        other_blocks = [b for b in content if b.get("type") != "tool_result"]

        messages: List[Dict[str, Any]] = []

        # First, handle tool results as separate tool messages
        for tool_result in tool_result_blocks:
            tool_content = tool_result.get("content", "")
            if isinstance(tool_content, list):
                # Extract text from content blocks
                text_parts = []
                for block in tool_content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                tool_content = "\n".join(text_parts)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result.get("tool_use_id", ""),
                    "content": str(tool_content) if tool_content else "",
                }
            )

        # Then handle remaining content
        if other_blocks:
            converted_content = _convert_anthropic_content_to_openai(other_blocks)
            messages.append({"role": role, "content": converted_content})
        elif not tool_result_blocks:
            # If there are no tool results and no other blocks, use empty string
            messages.append({"role": role, "content": ""})

        return messages if messages else [{"role": role, "content": ""}]

    # Simple string content
    return [{"role": role, "content": content}]


def _convert_anthropic_tools_to_openai(
    tools: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function format."""
    openai_tools = []

    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        openai_tools.append(openai_tool)

    return openai_tools


def _convert_anthropic_tool_choice_to_openai(
    tool_choice: Union[Dict[str, Any], str, None]
) -> Optional[Union[str, Dict[str, Any]]]:
    """Convert Anthropic tool_choice to OpenAI format."""
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    choice_type = tool_choice.get("type", "auto")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        tool_name = tool_choice.get("name")
        if tool_name:
            return {"type": "function", "function": {"name": tool_name}}
        return "auto"

    return "auto"


def request_to_openai(anthropic_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an Anthropic Messages API request to OpenAI Chat Completions format.

    Args:
        anthropic_request: Dictionary containing Anthropic request parameters

    Returns:
        Dictionary containing OpenAI request parameters
    """
    openai_request: Dict[str, Any] = {}

    # Model (pass through - the library handles model routing)
    openai_request["model"] = anthropic_request.get("model", "")

    # Convert messages
    anthropic_messages = anthropic_request.get("messages", [])
    openai_messages: List[Dict[str, Any]] = []

    # Handle system prompt
    system = anthropic_request.get("system")
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # System can be a list of text blocks
            system_text = " ".join(
                block.get("text", "") for block in system if block.get("type") == "text"
            )
            openai_messages.append({"role": "system", "content": system_text})

    # Convert each message
    for msg in anthropic_messages:
        converted = _convert_anthropic_message_to_openai(msg)
        openai_messages.extend(converted)

    openai_request["messages"] = openai_messages

    # max_tokens → max_completion_tokens (OpenAI's newer parameter name)
    # Also include max_tokens for compatibility with older models
    max_tokens = anthropic_request.get("max_tokens")
    if max_tokens is not None:
        openai_request["max_tokens"] = max_tokens

    # Temperature
    temperature = anthropic_request.get("temperature")
    if temperature is not None:
        openai_request["temperature"] = temperature

    # Top P
    top_p = anthropic_request.get("top_p")
    if top_p is not None:
        openai_request["top_p"] = top_p

    # Stop sequences
    stop_sequences = anthropic_request.get("stop_sequences")
    if stop_sequences:
        openai_request["stop"] = stop_sequences

    # Stream
    stream = anthropic_request.get("stream", False)
    openai_request["stream"] = stream

    # Tools
    tools = anthropic_request.get("tools")
    if tools:
        openai_request["tools"] = _convert_anthropic_tools_to_openai(tools)

    # Tool choice
    tool_choice = anthropic_request.get("tool_choice")
    if tool_choice is not None:
        converted_choice = _convert_anthropic_tool_choice_to_openai(tool_choice)
        if converted_choice is not None:
            openai_request["tool_choice"] = converted_choice

    return openai_request


# =============================================================================
# RESPONSE TRANSLATION: OpenAI → Anthropic
# =============================================================================


def _convert_openai_content_to_anthropic(
    content: Optional[str], tool_calls: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """Convert OpenAI response content to Anthropic content blocks."""
    blocks: List[Dict[str, Any]] = []

    # Add text content if present
    if content:
        blocks.append({"type": "text", "text": content})

    # Add tool use blocks if present
    if tool_calls:
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            arguments = function.get("arguments", "{}")

            # Parse arguments JSON
            import json

            try:
                parsed_args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed_args = {}

            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", _generate_id("toolu")),
                    "name": function.get("name", ""),
                    "input": parsed_args,
                }
            )

    return blocks if blocks else [{"type": "text", "text": ""}]


def _convert_openai_finish_reason_to_anthropic(
    finish_reason: Optional[str], has_tool_calls: bool = False
) -> Optional[str]:
    """Convert OpenAI finish_reason to Anthropic stop_reason."""
    if has_tool_calls:
        return "tool_use"

    if finish_reason is None:
        return None

    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",  # Anthropic doesn't have content_filter
        "function_call": "tool_use",  # Legacy function calling
    }

    return mapping.get(finish_reason, "end_turn")


def response_from_openai(
    openai_response: Dict[str, Any], original_model: str
) -> Dict[str, Any]:
    """
    Convert an OpenAI Chat Completions response to Anthropic Messages format.

    Args:
        openai_response: Dictionary containing OpenAI response
        original_model: The model name from the original Anthropic request

    Returns:
        Dictionary containing Anthropic response format
    """
    # Extract the first choice
    choices = openai_response.get("choices", [])
    choice = choices[0] if choices else {}

    message = choice.get("message", {})
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    finish_reason = choice.get("finish_reason")

    # Convert content
    anthropic_content = _convert_openai_content_to_anthropic(content, tool_calls)

    # Convert usage
    openai_usage = openai_response.get("usage", {})
    anthropic_usage = {
        "input_tokens": openai_usage.get("prompt_tokens", 0),
        "output_tokens": openai_usage.get("completion_tokens", 0),
    }

    # Handle cache tokens if present
    if "prompt_tokens_details" in openai_usage:
        details = openai_usage["prompt_tokens_details"]
        if "cached_tokens" in details:
            anthropic_usage["cache_read_input_tokens"] = details["cached_tokens"]

    # Build response
    anthropic_response = {
        "id": openai_response.get("id", _generate_id("msg")),
        "type": "message",
        "role": "assistant",
        "content": anthropic_content,
        "model": original_model,
        "stop_reason": _convert_openai_finish_reason_to_anthropic(
            finish_reason, bool(tool_calls)
        ),
        "stop_sequence": None,
        "usage": anthropic_usage,
    }

    return anthropic_response
