# Anthropic API Compatibility Guide

This document details how the proxy implements Anthropic's Messages API format and ensures compatibility with Anthropic clients like Claude Code.

## Table of Contents
- [API Endpoint](#api-endpoint)
- [Authentication](#authentication)
- [Request Format](#request-format)
- [Response Format](#response-format)
- [Streaming (SSE)](#streaming-sse)
- [Tool Calling](#tool-calling)
- [Error Handling](#error-handling)
- [Differences from Native Anthropic API](#differences-from-native-anthropic-api)

---

## API Endpoint

### POST /v1/messages

The main endpoint for sending messages and receiving responses.

**URL:** `http://localhost:8000/v1/messages`

**Method:** `POST`

**Headers:**
- `x-api-key: YOUR_PROXY_API_KEY` (required) - Your proxy's authentication key
- `anthropic-version: 2023-06-01` (optional) - API version for compatibility
- `Content-Type: application/json` (required)

**Alternative Authentication:**
- `Authorization: Bearer YOUR_PROXY_API_KEY` - Also supported for OpenAI compatibility

---

## Authentication

The proxy accepts Anthropic-style authentication:

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '...'
```

Or OpenAI-style:

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '...'
```

**Important:** Use your `PROXY_API_KEY` from the `.env` file, NOT your provider API keys.

---

## Request Format

### Basic Request

```json
{
  "model": "openai/gpt-4o",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": "Hello, Claude!"
    }
  ]
}
```

### Fields

#### Required Fields

- **model** (string): Model identifier in `provider/model-name` format
  - Examples: `openai/gpt-4o`, `gemini/gemini-2.0-flash-exp`, `anthropic/claude-3-5-sonnet-20241022`
  
- **max_tokens** (integer): Maximum tokens to generate
  - Range: 1 to model's limit (typically 4096-8192)
  - Required by Anthropic format (unlike OpenAI where it's optional)

- **messages** (array): Conversation history
  - Must alternate between `user` and `assistant` roles
  - First message must be from `user`

#### Optional Fields

- **system** (string or array): System prompt/instructions
  ```json
  {
    "system": "You are a helpful coding assistant.",
    "messages": [...]
  }
  ```
  
  Or with caching (Anthropic feature):
  ```json
  {
    "system": [
      {
        "type": "text",
        "text": "You are a helpful assistant.",
        "cache_control": {"type": "ephemeral"}
      }
    ],
    "messages": [...]
  }
  ```

- **temperature** (float): Sampling temperature (0.0 to 2.0)
  - Default: 1.0
  - Lower = more deterministic, Higher = more random

- **top_p** (float): Nucleus sampling threshold (0.0 to 1.0)
  - Default: 1.0

- **top_k** (integer): Top-k sampling parameter
  - Provider-specific

- **stream** (boolean): Enable streaming responses
  - Default: false

- **stop_sequences** (array): Custom stop sequences
  ```json
  {
    "stop_sequences": ["\n\nHuman:", "\n\nAssistant:"]
  }
  ```

- **tools** (array): Available tools for function calling
  - See [Tool Calling](#tool-calling) section

- **tool_choice** (object): Control tool selection
  ```json
  {
    "tool_choice": {
      "type": "tool",
      "name": "get_weather"
    }
  }
  ```

### Message Content Formats

#### Simple Text

```json
{
  "role": "user",
  "content": "Hello!"
}
```

#### Multi-Part Content

```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What's in this image?"
    },
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "base64_encoded_image_data"
      }
    }
  ]
}
```

#### Tool Result

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_123",
      "content": "The weather is sunny, 72°F"
    }
  ]
}
```

### Complete Example

```json
{
  "model": "openai/gpt-4o",
  "max_tokens": 2048,
  "temperature": 0.7,
  "system": "You are a helpful AI assistant specialized in Python programming.",
  "messages": [
    {
      "role": "user",
      "content": "Write a function to calculate factorial"
    },
    {
      "role": "assistant",
      "content": "Here's a factorial function:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"
    },
    {
      "role": "user",
      "content": "Now make it iterative"
    }
  ]
}
```

---

## Response Format

### Non-Streaming Response

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Hello! I'm Claude, an AI assistant created by Anthropic. How can I help you today?"
    }
  ],
  "model": "openai/gpt-4o",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25
  }
}
```

### Fields

- **id** (string): Unique message identifier (`msg_*`)
- **type** (string): Always "message"
- **role** (string): Always "assistant"
- **content** (array): Response content blocks
- **model** (string): Model used for generation
- **stop_reason** (string): Why generation stopped
  - `end_turn` - Natural completion
  - `max_tokens` - Hit token limit
  - `tool_use` - Model wants to call a tool
  - `stop_sequence` - Hit a stop sequence
- **stop_sequence** (string or null): The stop sequence that triggered stop
- **usage** (object): Token usage statistics
  - `input_tokens` - Input tokens consumed
  - `output_tokens` - Output tokens generated

### Response with Tool Use

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Let me check the weather for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_xyz789",
      "name": "get_weather",
      "input": {
        "location": "San Francisco, CA",
        "unit": "fahrenheit"
      }
    }
  ],
  "model": "openai/gpt-4o",
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 150,
    "output_tokens": 75
  }
}
```

---

## Streaming (SSE)

Enable streaming by setting `"stream": true` in the request.

### Event Types

The proxy sends Server-Sent Events (SSE) in Anthropic's format:

#### 1. message_start

Sent at the beginning of the stream.

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc123","type":"message","role":"assistant","content":[],"model":"openai/gpt-4o","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0}}}
```

#### 2. content_block_start

Sent when a new content block begins (text or tool_use).

```
event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
```

#### 3. content_block_delta

Sent for incremental content updates.

**Text Delta:**
```
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}
```

**Tool Input Delta:**
```
event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\""}}
```

#### 4. content_block_stop

Sent when a content block completes.

```
event: content_block_stop
data: {"type":"content_block_stop","index":0}
```

#### 5. message_delta

Sent when the message completes (contains stop_reason).

```
event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":25}}
```

#### 6. message_stop

Final event indicating stream completion.

```
event: message_stop
data: {"type":"message_stop"}
```

#### 7. error

Sent if an error occurs during streaming.

```
event: error
data: {"type":"error","error":{"type":"api_error","message":"Service temporarily unavailable"}}
```

### Streaming Example

**Request:**
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Count to 3"}],
    "stream": true
  }'
```

**Response:**
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc123",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"1"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\n"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"2"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\n"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"3"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}
```

### Streaming with Tool Use

When the model uses a tool, you'll see tool_use content blocks:

```
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_xyz","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\\"San"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" Francisco\\""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}
```

---

## Tool Calling

The proxy supports Anthropic's tool calling format (function calling).

### Defining Tools

```json
{
  "model": "openai/gpt-4o",
  "max_tokens": 1024,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather in a given location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The unit of temperature"
          }
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in SF?"
    }
  ]
}
```

### Tool Use Response

When the model decides to use a tool:

```json
{
  "id": "msg_abc123",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_xyz789",
      "name": "get_weather",
      "input": {
        "location": "San Francisco, CA",
        "unit": "fahrenheit"
      }
    }
  ],
  "stop_reason": "tool_use",
  ...
}
```

### Sending Tool Results

After executing the tool, send the result back:

```json
{
  "model": "openai/gpt-4o",
  "max_tokens": 1024,
  "tools": [...],  // Same tools array
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in SF?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "tool_use",
          "id": "toolu_xyz789",
          "name": "get_weather",
          "input": {
            "location": "San Francisco, CA",
            "unit": "fahrenheit"
          }
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_xyz789",
          "content": "The weather in San Francisco is currently 68°F and partly cloudy."
        }
      ]
    }
  ]
}
```

### Tool Choice

Control tool selection behavior:

**Auto (default):**
```json
{
  "tool_choice": {"type": "auto"}
}
```

**Force a specific tool:**
```json
{
  "tool_choice": {
    "type": "tool",
    "name": "get_weather"
  }
}
```

**Require any tool:**
```json
{
  "tool_choice": {"type": "any"}
}
```

### Complete Tool Calling Flow

```python
import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "YOUR_PROXY_API_KEY"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# Step 1: Initial request with tools
response = requests.post(
    f"{BASE_URL}/v1/messages",
    headers=headers,
    json={
        "model": "openai/gpt-4o",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Paris?"
            }
        ]
    }
)

result = response.json()
print("Assistant response:", json.dumps(result, indent=2))

# Step 2: Execute tool if requested
if result["stop_reason"] == "tool_use":
    tool_use = next(
        block for block in result["content"] 
        if block["type"] == "tool_use"
    )
    
    # Execute the tool (mock)
    tool_result = "The weather in Paris is 15°C and rainy."
    
    # Step 3: Send tool result back
    response = requests.post(
        f"{BASE_URL}/v1/messages",
        headers=headers,
        json={
            "model": "openai/gpt-4o",
            "max_tokens": 1024,
            "tools": [...],  # Same tools
            "messages": [
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": result["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": tool_result
                        }
                    ]
                }
            ]
        }
    )
    
    final_result = response.json()
    print("Final response:", json.dumps(final_result, indent=2))
```

---

## Error Handling

### Error Response Format

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "max_tokens is required"
  }
}
```

### Error Types

- **invalid_request_error** - Malformed request
  - Missing required fields
  - Invalid parameter values
  - Malformed JSON

- **authentication_error** - Invalid API key
  - Missing or incorrect `x-api-key` header
  - Expired proxy API key

- **rate_limit_error** - Rate limit exceeded
  - Provider rate limit hit
  - All credentials on cooldown

- **api_error** - Upstream API error
  - Provider service unavailable
  - Provider internal error
  - Network issues

### HTTP Status Codes

- **200 OK** - Success
- **400 Bad Request** - Invalid request
- **401 Unauthorized** - Invalid API key
- **429 Too Many Requests** - Rate limit exceeded
- **500 Internal Server Error** - Server error
- **502 Bad Gateway** - Upstream provider error
- **503 Service Unavailable** - Provider service unavailable
- **504 Gateway Timeout** - Request timeout

### Example Error Responses

**Missing Required Field:**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "max_tokens is required in Anthropic format"
  }
}
```

**Invalid API Key:**
```json
{
  "type": "error",
  "error": {
    "type": "authentication_error",
    "message": "Invalid or missing API Key. Use x-api-key header or Authorization: Bearer."
  }
}
```

**Rate Limit:**
```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "Rate Limit Exceeded: All credentials on cooldown"
  }
}
```

---

## Differences from Native Anthropic API

While the proxy strives for full compatibility, there are some differences:

### 1. Model Names

**Native Anthropic:**
```json
{"model": "claude-3-5-sonnet-20241022"}
```

**This Proxy:**
```json
{"model": "anthropic/claude-3-5-sonnet-20241022"}
```

The proxy requires provider prefix for routing. You can also use other providers:
```json
{"model": "openai/gpt-4o"}
{"model": "gemini/gemini-2.0-flash-exp"}
```

### 2. Available Models

The proxy supports ANY model from configured providers, not just Anthropic models.

List available models:
```bash
curl -H "x-api-key: YOUR_PROXY_API_KEY" \
     http://localhost:8000/v1/models
```

### 3. Provider-Specific Features

Some features may behave differently depending on the underlying provider:

- **Prompt Caching** - Only works with native Anthropic models
- **Vision** - Depends on provider support
- **Tool Calling** - Translated to provider's format (may have limitations)
- **Thinking/Reasoning** - Provider-specific (o1, Gemini thinking mode, etc.)

### 4. Rate Limits

Rate limits depend on YOUR provider API keys, not Anthropic's limits.

### 5. Pricing

Token usage and costs reflect the underlying provider, not Anthropic pricing.

Check costs:
```bash
curl -X POST http://localhost:8000/v1/cost-estimate \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "prompt_tokens": 1000,
    "completion_tokens": 500
  }'
```

### 6. System Messages

The proxy handles system messages correctly by extracting them from the messages array and passing them separately to providers that support it (OpenAI, Anthropic, etc.).

### 7. Additional Endpoints

The proxy provides extra endpoints not in Anthropic's API:

- `GET /v1/models` - List available models
- `GET /v1/providers` - List configured providers
- `POST /v1/token-count` - Calculate token counts
- `POST /v1/cost-estimate` - Estimate request cost
- `POST /v1/chat/completions` - OpenAI-compatible endpoint

---

## Testing Compatibility

### Test with cURL

```bash
# Basic test
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming test
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'

# Tool calling test
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "max_tokens": 1024,
    "tools": [{
      "name": "get_time",
      "description": "Get current time",
      "input_schema": {"type": "object", "properties": {}}
    }],
    "messages": [{"role": "user", "content": "What time is it?"}]
  }'
```

### Test with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/messages",
    headers={
        "x-api-key": "YOUR_PROXY_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "model": "openai/gpt-4o-mini",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json())
```

### Test with Claude Code

```bash
export ANTHROPIC_API_KEY="YOUR_PROXY_API_KEY"
export ANTHROPIC_BASE_URL="http://localhost:8000"

claude "Write a hello world in Python"
```

---

## Additional Resources

- **Setup Guide**: [CLAUDE_CODE_SETUP.md](CLAUDE_CODE_SETUP.md) - Complete Claude Code setup
- **Main README**: [README.md](README.md) - Feature overview
- **Technical Docs**: [DOCUMENTATION.md](DOCUMENTATION.md) - Architecture details
- **Anthropic API Docs**: https://docs.anthropic.com/en/api/messages

---

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/Mirrowel/LLM-API-Key-Proxy/issues
- **Enable Logging**: `--enable-request-logging` flag for debugging
- **Check Logs**: `logs/proxy.log` and `logs/detailed_logs/`

---

**Last Updated:** 2026-01-01
