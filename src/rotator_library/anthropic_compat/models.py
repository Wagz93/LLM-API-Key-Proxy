"""
Anthropic API Pydantic models.

This module contains all Pydantic models for Anthropic's Messages API,
including request/response structures for both streaming and non-streaming calls.

These models are framework-agnostic and can be used independently of FastAPI.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# =============================================================================
# CONTENT BLOCKS
# =============================================================================


class TextBlock(BaseModel):
    """Anthropic text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    """Image source for Anthropic vision."""

    type: Literal["base64", "url"] = "base64"
    media_type: str
    data: str


class ImageBlock(BaseModel):
    """Anthropic image content block."""

    type: Literal["image"] = "image"
    source: ImageSource


class ToolUseBlock(BaseModel):
    """Anthropic tool use content block (in assistant messages)."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultBlock(BaseModel):
    """Anthropic tool result content block (in user messages)."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, List[Union["TextBlock", "ImageBlock"]]]] = None
    is_error: Optional[bool] = None


# Content block union type
ContentBlock = Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock]


# =============================================================================
# MESSAGES
# =============================================================================


class AnthropicMessage(BaseModel):
    """An Anthropic message with role and content."""

    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


# =============================================================================
# TOOLS
# =============================================================================


class ToolInputSchema(BaseModel):
    """JSON Schema for tool input parameters."""

    type: Literal["object"] = "object"
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: Optional[List[str]] = None


class AnthropicTool(BaseModel):
    """Anthropic tool definition."""

    name: str
    description: Optional[str] = None
    input_schema: ToolInputSchema


class ToolChoice(BaseModel):
    """Anthropic tool choice specification."""

    type: Literal["auto", "any", "tool"]
    name: Optional[str] = None  # Only for type="tool"
    disable_parallel_tool_use: Optional[bool] = None


# =============================================================================
# METADATA
# =============================================================================


class AnthropicMetadata(BaseModel):
    """Anthropic request metadata."""

    user_id: Optional[str] = None


# =============================================================================
# REQUEST
# =============================================================================


class AnthropicMessagesRequest(BaseModel):
    """
    Anthropic Messages API request format.

    This model represents the full request structure for Anthropic's /v1/messages endpoint.
    """

    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, List[TextBlock]]] = None
    metadata: Optional[AnthropicMetadata] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Union[ToolChoice, Dict[str, Any]]] = None


# =============================================================================
# RESPONSE
# =============================================================================


class Usage(BaseModel):
    """Token usage information."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicMessagesResponse(BaseModel):
    """
    Anthropic Messages API response format.

    This model represents the full response structure from Anthropic's /v1/messages endpoint.
    """

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock]
    model: str
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# =============================================================================
# STREAMING EVENTS
# =============================================================================


class MessageStartEvent(BaseModel):
    """message_start event data."""

    type: Literal["message_start"] = "message_start"
    message: Dict[str, Any]  # Partial message object


class ContentBlockStartEvent(BaseModel):
    """content_block_start event data."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: Dict[str, Any]


class ContentBlockDeltaEvent(BaseModel):
    """content_block_delta event data."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Dict[str, Any]


class ContentBlockStopEvent(BaseModel):
    """content_block_stop event data."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """message_delta event data."""

    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None


class MessageStopEvent(BaseModel):
    """message_stop event data."""

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """ping event data."""

    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """error event data."""

    type: Literal["error"] = "error"
    error: Dict[str, Any]


# Union of all streaming event types
StreamingEvent = Union[
    MessageStartEvent,
    ContentBlockStartEvent,
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent,
    PingEvent,
    ErrorEvent,
]


# =============================================================================
# TOKEN COUNTING
# =============================================================================


class CountTokensRequest(BaseModel):
    """Anthropic count_tokens request format."""

    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[TextBlock]]] = None
    tools: Optional[List[AnthropicTool]] = None


class CountTokensResponse(BaseModel):
    """Anthropic count_tokens response format."""

    input_tokens: int
