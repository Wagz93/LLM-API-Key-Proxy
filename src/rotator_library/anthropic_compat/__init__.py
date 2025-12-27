"""
Anthropic Compatibility Layer for the LLM Routing Library.

This module provides a translation layer that allows Anthropic API clients
to work with the OpenAI-compatible internal format used by the library.

The layer handles:
- Request translation: Anthropic Messages API → OpenAI Chat Completions
- Response translation: OpenAI Chat Completions → Anthropic Messages API
- Streaming conversion: OpenAI SSE format → Anthropic SSE format

Usage:
    The compatibility layer can be used in two ways:

    1. Through the RotatingClient's anthropic_messages() method:
        response = await client.anthropic_messages(request_data)

    2. Directly using the translator functions:
        from rotator_library.anthropic_compat import (
            request_to_openai,
            response_from_openai,
            convert_openai_stream_to_anthropic,
        )
"""

from .models import (
    # Request/Response models
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicMessage,
    AnthropicMetadata,
    # Content blocks
    TextBlock,
    ImageBlock,
    ImageSource,
    ToolUseBlock,
    ToolResultBlock,
    ContentBlock,
    # Tools
    AnthropicTool,
    ToolInputSchema,
    ToolChoice,
    # Usage
    Usage,
    # Streaming events
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent,
    StreamingEvent,
    # Token counting
    CountTokensRequest,
    CountTokensResponse,
)

from .translator import (
    request_to_openai,
    response_from_openai,
)

from .streaming import (
    convert_openai_stream_to_anthropic,
    create_error_event,
    StreamingState,
)

__all__ = [
    # Request/Response models
    "AnthropicMessagesRequest",
    "AnthropicMessagesResponse",
    "AnthropicMessage",
    "AnthropicMetadata",
    # Content blocks
    "TextBlock",
    "ImageBlock",
    "ImageSource",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Tools
    "AnthropicTool",
    "ToolInputSchema",
    "ToolChoice",
    # Usage
    "Usage",
    # Streaming events
    "MessageStartEvent",
    "ContentBlockStartEvent",
    "ContentBlockDeltaEvent",
    "ContentBlockStopEvent",
    "MessageDeltaEvent",
    "MessageStopEvent",
    "PingEvent",
    "ErrorEvent",
    "StreamingEvent",
    # Token counting
    "CountTokensRequest",
    "CountTokensResponse",
    # Translator functions
    "request_to_openai",
    "response_from_openai",
    # Streaming functions
    "convert_openai_stream_to_anthropic",
    "create_error_event",
    "StreamingState",
]
