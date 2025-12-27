"""
Anthropic ↔ OpenAI format translator.

This module provides bidirectional translation between Anthropic's Messages API format
and OpenAI's Chat Completions API format.

The translation layer is framework-agnostic and operates on dictionaries,
making it usable in any Python context.

Translation Flow:
1. Anthropic Request → OpenAI Request (request_to_openai)
2. OpenAI Response → Anthropic Response (response_from_openai)

Security Notes:
- All inputs are validated before processing
- String lengths are bounded to prevent DoS
- JSON parsing uses strict mode with error handling
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union

# =============================================================================
# CONSTANTS & LIMITS
# =============================================================================

# Maximum allowed values for safety
MAX_TOKENS_LIMIT = 1_000_000  # Reasonable upper bound for any model
MAX_STRING_LENGTH = 10_000_000  # 10MB max for any single string field
MAX_MESSAGES = 10_000  # Maximum number of messages in a conversation
MAX_TOOLS = 1_000  # Maximum number of tools


# =============================================================================
# UTILITIES
# =============================================================================


def _generate_id(prefix: str = "msg") -> str:
    """Generate a unique ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _validate_string(value: Any, field_name: str, max_length: int = MAX_STRING_LENGTH) -> str:
    """Validate and bound a string value."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    if len(value) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length of {max_length}")
    return value


def _validate_positive_int(value: Any, field_name: str, max_value: int = MAX_TOKENS_LIMIT) -> Optional[int]:
    """Validate a positive integer value."""
    if value is None:
        return None
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be an integer")
    if int_value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    if int_value > max_value:
        raise ValueError(f"{field_name} exceeds maximum value of {max_value}")
    return int_value


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
    
    Handles:
    - Simple text messages
    - Messages with tool_result blocks (user role) -> tool role
    - Messages with tool_use blocks (assistant role) -> tool_calls
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")

    # Handle list content (can contain various block types)
    if isinstance(content, list):
        # Single pass categorization for performance
        tool_result_blocks: List[Dict[str, Any]] = []
        tool_use_blocks: List[Dict[str, Any]] = []
        text_blocks: List[Dict[str, Any]] = []
        other_blocks: List[Dict[str, Any]] = []
        
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                tool_result_blocks.append(block)
            elif block_type == "tool_use":
                tool_use_blocks.append(block)
            elif block_type == "text":
                text_blocks.append(block)
            else:
                other_blocks.append(block)

        messages: List[Dict[str, Any]] = []

        # Handle tool results as separate tool messages (these come from user role)
        for tool_result in tool_result_blocks:
            tool_content = tool_result.get("content", "")
            if isinstance(tool_content, list):
                # Extract text from content blocks
                text_parts = []
                for block in tool_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
                tool_content = "\n".join(text_parts)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_result.get("tool_use_id", "")),
                    "content": str(tool_content) if tool_content else "",
                }
            )

        # Handle assistant messages with tool_use blocks
        if role == "assistant" and tool_use_blocks:
            # Extract text content if any
            text_content = ""
            if text_blocks:
                text_content = " ".join(
                    str(b.get("text", "")) for b in text_blocks
                )
            
            # Convert tool_use blocks to OpenAI tool_calls
            tool_calls = []
            for i, tool_use in enumerate(tool_use_blocks):
                tool_input = tool_use.get("input", {})
                # Ensure input is serialized as JSON string
                if isinstance(tool_input, dict):
                    arguments = json.dumps(tool_input)
                else:
                    arguments = str(tool_input) if tool_input else "{}"
                
                tool_calls.append({
                    "id": str(tool_use.get("id", _generate_id("call"))),
                    "type": "function",
                    "function": {
                        "name": str(tool_use.get("name", "")),
                        "arguments": arguments,
                    }
                })
            
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            if text_content:
                assistant_msg["content"] = text_content
            else:
                assistant_msg["content"] = None
            assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)
        
        # Handle remaining text/other blocks for non-assistant or non-tool-use cases
        elif text_blocks or other_blocks:
            blocks_to_convert = text_blocks + other_blocks
            converted_content = _convert_anthropic_content_to_openai(blocks_to_convert)
            messages.append({"role": role, "content": converted_content})
        
        # If no content at all (edge case)
        elif not tool_result_blocks and not tool_use_blocks:
            messages.append({"role": role, "content": ""})

        return messages if messages else [{"role": role, "content": ""}]

    # Simple string content
    return [{"role": role, "content": str(content) if content else ""}]


def _convert_anthropic_tools_to_openai(
    tools: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert Anthropic tool definitions to OpenAI function format.
    
    Validates tool names and structures.
    """
    openai_tools = []

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Tool at index {i} must be a dictionary")
        
        name = tool.get("name")
        if not name:
            raise ValueError(f"Tool at index {i} is missing required 'name' field")
        
        # Validate tool name length (max 64 chars per OpenAI spec)
        name_str = str(name)
        if len(name_str) > 64:
            raise ValueError(f"Tool name '{name_str[:20]}...' exceeds maximum length of 64")
        
        description = tool.get("description", "")
        input_schema = tool.get("input_schema", {"type": "object"})
        
        # Ensure input_schema is a valid object
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object"}
        
        openai_tool = {
            "type": "function",
            "function": {
                "name": name_str,
                "description": str(description) if description else "",
                "parameters": input_schema,
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

    Raises:
        ValueError: If required fields are missing or invalid
    """
    if not isinstance(anthropic_request, dict):
        raise ValueError("Request must be a dictionary")

    openai_request: Dict[str, Any] = {}

    # Model (required, pass through - the library handles model routing)
    model = anthropic_request.get("model")
    if not model:
        raise ValueError("'model' is required")
    openai_request["model"] = _validate_string(model, "model", max_length=256)

    # Validate and convert messages
    anthropic_messages = anthropic_request.get("messages", [])
    if not isinstance(anthropic_messages, list):
        raise ValueError("'messages' must be a list")
    if len(anthropic_messages) > MAX_MESSAGES:
        raise ValueError(f"'messages' exceeds maximum count of {MAX_MESSAGES}")

    openai_messages: List[Dict[str, Any]] = []

    # Handle system prompt
    system = anthropic_request.get("system")
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": _validate_string(system, "system")})
        elif isinstance(system, list):
            # System can be a list of text blocks
            system_text = " ".join(
                _validate_string(block.get("text", ""), "system.text")
                for block in system if isinstance(block, dict) and block.get("type") == "text"
            )
            if system_text:
                openai_messages.append({"role": "system", "content": system_text})

    # Convert each message
    for msg in anthropic_messages:
        if not isinstance(msg, dict):
            raise ValueError("Each message must be a dictionary")
        converted = _convert_anthropic_message_to_openai(msg)
        openai_messages.extend(converted)

    openai_request["messages"] = openai_messages

    # max_tokens (required for Anthropic, validated)
    max_tokens = anthropic_request.get("max_tokens")
    if max_tokens is not None:
        validated_max_tokens = _validate_positive_int(max_tokens, "max_tokens")
        if validated_max_tokens is not None:
            openai_request["max_tokens"] = validated_max_tokens

    # Temperature: Anthropic supports 0.0-1.0, OpenAI supports 0.0-2.0
    # We accept the wider OpenAI range since we're routing to various backends
    temperature = anthropic_request.get("temperature")
    if temperature is not None:
        try:
            temp_float = float(temperature)
            if not (0.0 <= temp_float <= 2.0):
                raise ValueError("'temperature' must be between 0.0 and 2.0")
            openai_request["temperature"] = temp_float
        except (TypeError, ValueError) as e:
            if "must be between" in str(e):
                raise
            raise ValueError("'temperature' must be a number")

    # Top P (0.0 to 1.0)
    top_p = anthropic_request.get("top_p")
    if top_p is not None:
        try:
            top_p_float = float(top_p)
            if not (0.0 <= top_p_float <= 1.0):
                raise ValueError("'top_p' must be between 0.0 and 1.0")
            openai_request["top_p"] = top_p_float
        except (TypeError, ValueError) as e:
            if "must be between" in str(e):
                raise
            raise ValueError("'top_p' must be a number")

    # Stop sequences
    stop_sequences = anthropic_request.get("stop_sequences")
    if stop_sequences:
        if not isinstance(stop_sequences, list):
            raise ValueError("'stop_sequences' must be a list")
        validated_sequences = [_validate_string(s, "stop_sequence", max_length=1000) for s in stop_sequences]
        openai_request["stop"] = validated_sequences

    # Stream
    stream = anthropic_request.get("stream", False)
    openai_request["stream"] = bool(stream)

    # Tools
    tools = anthropic_request.get("tools")
    if tools:
        if not isinstance(tools, list):
            raise ValueError("'tools' must be a list")
        if len(tools) > MAX_TOOLS:
            raise ValueError(f"'tools' exceeds maximum count of {MAX_TOOLS}")
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
