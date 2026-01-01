# Claude Code Setup Guide

Complete guide for using this LLM API Key Proxy with Claude Code (Anthropic's code assistant).

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Configuration](#configuration)
- [Testing Your Setup](#testing-your-setup)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Overview

This proxy transforms API requests to make any LLM provider (OpenAI, Google Gemini, etc.) appear as an Anthropic API endpoint. This allows you to use Claude Code with different LLM providers while maintaining full compatibility with Anthropic's Messages API format.

**What This Proxy Does:**
- Exposes `/v1/messages` endpoint (Anthropic Messages API)
- Accepts `x-api-key` header (Anthropic authentication style)
- Translates requests/responses between Anthropic and OpenAI formats
- Supports streaming with proper Anthropic SSE events
- Handles tool calling (function calls) in Anthropic format
- Manages multiple provider credentials with rotation

---

## Prerequisites

**Required:**
- Python 3.8 or higher
- Internet connection
- API keys for at least one LLM provider (OpenAI, Gemini, etc.)

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5, etc.)
- Google Gemini (via API key or OAuth)
- Anthropic (native support)
- OpenRouter
- Groq
- Mistral AI
- And many more...

---

## Quick Start

### 1. Install the Proxy

```bash
# Clone the repository
git clone https://github.com/Mirrowel/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
# Run the interactive setup tool
python src/proxy_app/main.py --add-credential
```

The credential tool will:
1. Create a `.env` file with a secure `PROXY_API_KEY`
2. Guide you through adding LLM provider credentials
3. Support both API keys and OAuth authentication

**Quick Manual Setup (.env file):**
```env
# Your proxy's API key (clients must use this)
PROXY_API_KEY="sk-your-secret-proxy-key-here"

# Add your LLM provider API keys
OPENAI_API_KEY_1="sk-..."
GEMINI_API_KEY_1="AIza..."
ANTHROPIC_API_KEY_1="sk-ant-..."
```

### 3. Start the Proxy

```bash
# Start on default port 8000
python src/proxy_app/main.py

# Or specify custom host/port
python src/proxy_app/main.py --host 127.0.0.1 --port 9000
```

You should see:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Starting proxy on 0.0.0.0:8000
Proxy API Key: ✓ sk-your-secret-proxy-key-here
GitHub: https://github.com/Mirrowel/LLM-API-Key-Proxy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Server ready in 2.34s (15 providers discovered in 0.45s)
```

### 4. Configure Claude Code

**Option A: Environment Variables**
```bash
export ANTHROPIC_API_KEY="sk-your-secret-proxy-key-here"
export ANTHROPIC_BASE_URL="http://localhost:8000"
```

**Option B: Configuration File**

Create or edit `~/.config/claude/config.json`:
```json
{
  "api_key": "sk-your-secret-proxy-key-here",
  "base_url": "http://localhost:8000"
}
```

**Note:** The `api_key` you provide to Claude Code should be your **PROXY_API_KEY** from the `.env` file, NOT your actual Anthropic or other provider API keys.

### 5. Test Claude Code

```bash
# Test with a simple command
claude "Hello, how are you?"

# Or use Claude Code in your IDE
# It will automatically use the configured proxy
```

---

## Detailed Setup

### Understanding the Architecture

```
┌─────────────┐    Anthropic    ┌─────────────┐   OpenAI/Gemini   ┌──────────────┐
│ Claude Code │───Messages API──▶│  This Proxy │────API Format────▶│ LLM Provider │
│             │◀──────SSE────────│             │◀──────JSON───────│  (Multiple)  │
└─────────────┘                  └─────────────┘                  └──────────────┘
                                       │
                                       ├─ Credential Rotation
                                       ├─ Error Handling
                                       ├─ Rate Limiting
                                       └─ Usage Tracking
```

**Key Components:**
1. **Anthropic Translator** - Converts between Anthropic and OpenAI formats
2. **Streaming Converter** - Transforms SSE events to Anthropic format
3. **Tool Handler** - Manages function calling in Anthropic format
4. **Credential Manager** - Rotates between multiple API keys
5. **Request Logger** - Detailed logging for debugging

### Setting Up Provider Credentials

#### API Key Based Providers

Add to `.env`:
```env
# OpenAI
OPENAI_API_KEY_1="sk-..."
OPENAI_API_KEY_2="sk-..."  # Multiple keys for rotation

# Google Gemini
GEMINI_API_KEY_1="AIza..."

# Anthropic (if you want to proxy actual Anthropic keys)
ANTHROPIC_API_KEY_1="sk-ant-..."

# OpenRouter (access 100+ models)
OPENROUTER_API_KEY_1="sk-or-..."

# Groq (fast inference)
GROQ_API_KEY_1="gsk_..."
```

#### OAuth Based Providers

Some providers support OAuth authentication for enhanced features:

```bash
# Run the credential tool
python src/proxy_app/main.py --add-credential

# Select "Add OAuth Credential"
# Choose your provider:
#   - Gemini CLI (Google OAuth)
#   - Antigravity (Gemini 3, Claude Opus 4.5)
#   - Qwen Code
#   - iFlow

# Follow the browser authentication flow
```

**OAuth Advantages:**
- Higher rate limits (Gemini CLI)
- Access to exclusive models (Antigravity: Gemini 3, Claude Opus 4.5)
- Zero-configuration project discovery
- Automatic credential refresh

### Model Selection

When making requests through Claude Code via this proxy, specify models in the format:

```
provider/model-name
```

**Examples:**
- `openai/gpt-4o` - OpenAI GPT-4 Optimized
- `gemini/gemini-2.0-flash-exp` - Google Gemini 2.0 Flash
- `anthropic/claude-3-5-sonnet-20241022` - Anthropic Claude 3.5 Sonnet
- `openrouter/anthropic/claude-3.5-sonnet` - Claude via OpenRouter
- `groq/llama-3.3-70b-versatile` - Llama 3.3 on Groq

**List Available Models:**
```bash
# Using curl
curl -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
     http://localhost:8000/v1/models

# Or visit in browser
http://localhost:8000/v1/models?enriched=true
```

---

## Configuration

### Proxy Settings

Key environment variables in `.env`:

```env
# [REQUIRED] Your proxy's authentication key
PROXY_API_KEY="sk-your-secret-proxy-key-here"

# [OPTIONAL] Concurrent requests per credential
MAX_CONCURRENT_REQUESTS_PER_KEY_OPENAI=3
MAX_CONCURRENT_REQUESTS_PER_KEY_GEMINI=1

# [OPTIONAL] Credential rotation mode
# balanced - Distribute load evenly (default)
# sequential - Use until exhausted (good for daily quotas)
ROTATION_MODE_OPENAI=balanced
ROTATION_MODE_GEMINI=sequential

# [OPTIONAL] Model filtering
# Hide specific models
IGNORE_MODELS_OPENAI="*-preview*"
# Or use whitelist mode (only show these)
IGNORE_MODELS_GEMINI="*"
WHITELIST_MODELS_GEMINI="gemini-2.0-flash-exp,gemini-1.5-pro"

# [OPTIONAL] Skip OAuth validation (for Docker/CI)
SKIP_OAUTH_INIT_CHECK=false

# [OPTIONAL] Enable detailed request logging
# When enabled, logs are saved to logs/detailed_logs/
# Note: Start with --enable-request-logging flag instead
```

### Claude Code Configuration

Claude Code looks for configuration in these locations (in order):

1. **Environment Variables** (highest priority)
   ```bash
   export ANTHROPIC_API_KEY="sk-your-secret-proxy-key-here"
   export ANTHROPIC_BASE_URL="http://localhost:8000"
   ```

2. **Config File**: `~/.config/claude/config.json`
   ```json
   {
     "api_key": "sk-your-secret-proxy-key-here",
     "base_url": "http://localhost:8000",
     "model": "openai/gpt-4o"
   }
   ```

3. **Project-Specific**: `.claude/config.json` in your project directory
   ```json
   {
     "api_key": "sk-your-secret-proxy-key-here",
     "base_url": "http://localhost:8000",
     "model": "gemini/gemini-2.0-flash-exp"
   }
   ```

**Important:** Always use your `PROXY_API_KEY` value, not your actual provider API keys, when configuring Claude Code.

### Advanced Proxy Configuration

#### Timeout Settings

Fine-tune HTTP timeouts for different scenarios:

```env
TIMEOUT_CONNECT=30              # Connection establishment
TIMEOUT_WRITE=30                # Request body send
TIMEOUT_POOL=60                 # Connection pool acquisition
TIMEOUT_READ_STREAMING=180      # Between streaming chunks
TIMEOUT_READ_NON_STREAMING=600  # Full response wait
```

**Recommendations:**
- Long thinking tasks: Increase `TIMEOUT_READ_STREAMING` to 300-360s
- Unstable network: Increase `TIMEOUT_CONNECT` to 60s
- Large outputs: Increase `TIMEOUT_READ_NON_STREAMING` to 900s+

#### Priority-Based Concurrency

Assign priority tiers to credentials for better resource management:

```env
# Priority 1 (paid/premium): 10x concurrency
CONCURRENCY_MULTIPLIER_ANTIGRAVITY_PRIORITY_1=10

# Priority 2 (standard paid): 3x
CONCURRENCY_MULTIPLIER_ANTIGRAVITY_PRIORITY_2=3
```

---

## Testing Your Setup

### Test 1: Health Check

```bash
curl http://localhost:8000/
```

Expected response:
```json
{"Status":"API Key Proxy is running"}
```

### Test 2: List Models

```bash
curl -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
     http://localhost:8000/v1/models
```

Should return a list of available models from all configured providers.

### Test 3: Simple Message (Non-Streaming)

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 100,
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'
```

Expected response (Anthropic format):
```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'm doing well, thank you! How can I help you today?"
    }
  ],
  "model": "openai/gpt-4o-mini",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 15
  }
}
```

### Test 4: Streaming Message

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

Expected output (SSE format):
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"1"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\n2"}}

...

event: message_stop
data: {"type":"message_stop"}
```

### Test 5: Tool Calling (Function Calling)

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o",
    "max_tokens": 1024,
    "tools": [
      {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state"
            }
          },
          "required": ["location"]
        }
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in San Francisco?"
      }
    ]
  }'
```

Expected response includes tool use:
```json
{
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_...",
      "name": "get_weather",
      "input": {
        "location": "San Francisco, CA"
      }
    }
  ],
  "stop_reason": "tool_use",
  ...
}
```

### Test 6: Claude Code Integration

```bash
# Set up environment
export ANTHROPIC_API_KEY="sk-your-secret-proxy-key-here"
export ANTHROPIC_BASE_URL="http://localhost:8000"

# Test with Claude Code
claude "Write a Python function to calculate fibonacci numbers"
```

---

## Troubleshooting

### Common Issues

#### Issue: "401 Unauthorized" Error

**Cause:** Incorrect or missing `PROXY_API_KEY`.

**Solution:**
1. Check your `.env` file has `PROXY_API_KEY` set
2. Ensure Claude Code is using the same key:
   ```bash
   echo $ANTHROPIC_API_KEY
   # Should match PROXY_API_KEY from .env
   ```
3. Restart the proxy after changing `.env`

#### Issue: "No Provider Credentials Configured"

**Cause:** No LLM provider API keys found.

**Solution:**
1. Run the credential tool:
   ```bash
   python src/proxy_app/main.py --add-credential
   ```
2. Or manually add keys to `.env`:
   ```env
   OPENAI_API_KEY_1="sk-..."
   ```
3. Restart the proxy

#### Issue: "Model not found"

**Cause:** Model name format incorrect or provider not configured.

**Solution:**
1. Use correct format: `provider/model-name`
   - ✅ `openai/gpt-4o`
   - ❌ `gpt-4o`
2. List available models:
   ```bash
   curl -H "Authorization: Bearer YOUR_PROXY_API_KEY" \
        http://localhost:8000/v1/models
   ```
3. Ensure provider credentials are configured

#### Issue: Streaming Hangs or Timeouts

**Cause:** Timeout settings too low for streaming.

**Solution:**
Add to `.env`:
```env
TIMEOUT_READ_STREAMING=300  # 5 minutes
```

Restart the proxy.

#### Issue: "All keys on cooldown"

**Cause:** All provider credentials hit rate limits.

**Solution:**
1. Check logs: `logs/detailed_logs/`
2. Add more API keys for rotation
3. Adjust rotation mode:
   ```env
   ROTATION_MODE_OPENAI=balanced  # Better distribution
   ```

#### Issue: OAuth Callback Failed

**Cause:** OAuth callback port blocked by firewall.

**Solution:**
1. Ensure ports are open:
   - Gemini CLI: 8085
   - Antigravity: 51121
   - iFlow: 11451
2. Or use SSH port forwarding:
   ```bash
   ssh -L 8085:localhost:8085 user@your-server
   ```

### Debugging Tips

#### Enable Detailed Logging

Start the proxy with logging enabled:
```bash
python src/proxy_app/main.py --enable-request-logging
```

Logs are saved to:
- `logs/proxy.log` - General logs
- `logs/proxy_debug.log` - Debug logs
- `logs/detailed_logs/` - Full request/response logs

#### Check Request/Response Flow

When detailed logging is enabled, check `logs/detailed_logs/`:
- `request.json` - Original request payload
- `final_response.json` - Complete response
- `streaming_chunks.jsonl` - All SSE chunks (for streaming)
- `metadata.json` - Performance metrics

#### Test Without Claude Code

Use `curl` to isolate issues:
```bash
# Test the /v1/messages endpoint directly
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Hi"}]
  }'
```

#### Check Credential Status

View detailed credential information:
```bash
python -m rotator_library.credential_tool
# Select "View Credentials"
```

### Getting Help

If you're still having issues:

1. **Check logs**: `logs/proxy.log` and `logs/proxy_debug.log`
2. **Enable request logging**: `--enable-request-logging`
3. **Search existing issues**: [GitHub Issues](https://github.com/Mirrowel/LLM-API-Key-Proxy/issues)
4. **Open a new issue** with:
   - Error messages from logs
   - Your configuration (sanitized, no API keys!)
   - Steps to reproduce
   - Proxy version/commit

---

## Advanced Usage

### Running Behind Nginx

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Important for streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

Then configure Claude Code:
```bash
export ANTHROPIC_BASE_URL="https://api.yourdomain.com"
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use environment variables for configuration
ENV SKIP_OAUTH_INIT_CHECK=true

EXPOSE 8000
CMD ["uvicorn", "src.proxy_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
# Build
docker build -t llm-proxy .

# Run with environment file
docker run -p 8000:8000 --env-file .env llm-proxy
```

### Multiple Model Strategies

Configure Claude Code to use different models for different tasks:

**Project A** (Fast responses):
```json
// .claude/config.json
{
  "model": "gemini/gemini-2.0-flash-exp",
  "base_url": "http://localhost:8000"
}
```

**Project B** (High quality):
```json
// .claude/config.json
{
  "model": "openai/gpt-4o",
  "base_url": "http://localhost:8000"
}
```

### Load Balancing Across Multiple Proxies

Run multiple proxy instances:
```bash
# Terminal 1
python src/proxy_app/main.py --port 8000

# Terminal 2
python src/proxy_app/main.py --port 8001

# Terminal 3
python src/proxy_app/main.py --port 8002
```

Use a load balancer (nginx, HAProxy) or configure Claude Code to round-robin between them.

### Stateless Deployment (Railway, Render, Vercel)

For platforms without file persistence:

1. Set up credentials locally:
   ```bash
   python -m rotator_library.credential_tool
   ```

2. Export to environment variables:
   ```bash
   python -m rotator_library.credential_tool
   # Select "Export [Provider] to .env"
   ```

3. Copy generated variables to your platform's environment settings

4. Set `SKIP_OAUTH_INIT_CHECK=true`

See [Deployment Guide](Deployment%20guide.md) for complete instructions.

---

## FAQ

**Q: Can I use actual Anthropic API keys through this proxy?**  
A: Yes! Add `ANTHROPIC_API_KEY_1="sk-ant-..."` to your `.env`. The proxy will use native Anthropic APIs for Anthropic models and translate for other providers.

**Q: What's the difference between PROXY_API_KEY and provider API keys?**  
A: `PROXY_API_KEY` is for authenticating to YOUR proxy. Provider API keys (OpenAI, Gemini, etc.) are what the proxy uses to call actual LLM services.

**Q: Does this work with Claude Code's tool calling features?**  
A: Yes! The proxy fully supports Anthropic's tool calling format and translates it to/from OpenAI's function calling format.

**Q: Can I use multiple providers simultaneously?**  
A: Yes! Add API keys for multiple providers. The proxy will route requests based on the model name (e.g., `openai/gpt-4` vs `gemini/gemini-2.0-flash`).

**Q: How do I update to the latest version?**  
A: 
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

**Q: Is there a rate limit on the proxy itself?**  
A: No, the proxy doesn't impose its own rate limits. You're only limited by your provider API keys' rate limits.

**Q: Can Claude Code use streaming with this proxy?**  
A: Yes! The proxy fully supports Server-Sent Events (SSE) streaming in Anthropic's format.

**Q: What happens if one of my API keys hits a rate limit?**  
A: The proxy automatically rotates to the next available key for that provider. The failed key is put on cooldown temporarily.

---

## Additional Resources

- **Main README**: [README.md](README.md) - Overview and feature list
- **Technical Documentation**: [DOCUMENTATION.md](DOCUMENTATION.md) - Architecture and internals
- **Deployment Guide**: [Deployment guide.md](Deployment%20guide.md) - Hosting on Render, Railway, VPS
- **Library README**: [src/rotator_library/README.md](src/rotator_library/README.md) - Using the library directly
- **GitHub Repository**: https://github.com/Mirrowel/LLM-API-Key-Proxy
- **Anthropic API Documentation**: https://docs.anthropic.com/en/api

---

## License

This project is dual-licensed:
- **Proxy Application** (`src/proxy_app/`) - MIT License
- **Resilience Library** (`src/rotator_library/`) - LGPL-3.0

See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a clear description

For bugs, please open an issue with:
- Error messages and logs
- Steps to reproduce
- Your environment (OS, Python version, etc.)

---

**Last Updated:** 2026-01-01  
**Proxy Version:** Compatible with all versions supporting `/v1/messages` endpoint
