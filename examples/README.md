# Examples

This directory contains example scripts and test utilities for the LLM API Key Proxy.

## Files

### anthropic_test.py

**Purpose:** Comprehensive test suite for Anthropic Messages API compatibility

**Tests:**
1. Health check - Verify proxy is running
2. List models - Check available models
3. Basic message - Non-streaming request/response
4. Streaming - SSE streaming with proper event format
5. Tool calling - Function calling in Anthropic format
6. Authentication - API key validation
7. Error handling - Invalid requests

**Usage:**
```bash
# Basic usage (uses PROXY_API_KEY from environment)
python examples/anthropic_test.py

# With explicit API key
python examples/anthropic_test.py --api-key YOUR_PROXY_API_KEY

# Custom URL and model
python examples/anthropic_test.py \
  --base-url http://localhost:8000 \
  --api-key YOUR_KEY \
  --model gemini/gemini-2.0-flash-exp
```

**Requirements:**
- `requests` library (installed with proxy requirements)
- Running proxy instance
- Valid PROXY_API_KEY

**Output:**
- Color-coded test results
- Detailed validation of responses
- Pass/fail summary

**Example Output:**
```
╔═══════════════════════════════════════════════════════════════════╗
║        Anthropic API Compatibility Test Suite                    ║
╚═══════════════════════════════════════════════════════════════════╝

Base URL: http://localhost:8000
Model: openai/gpt-4o-mini
API Key: sk-prox...

======================================================================
Test 1: Health Check
======================================================================

✓ Proxy is running

======================================================================
Test 2: List Models
======================================================================

✓ Found 45 available models
First 5 models: ['openai/gpt-4o', 'openai/gpt-4o-mini', ...]
✓ Test model 'openai/gpt-4o-mini' is available

...

======================================================================
Test Summary
======================================================================

Passed: 15
Failed: 0
Total: 15
Pass Rate: 100.0%

✓ All tests passed!
```

## Adding Your Own Examples

Feel free to add your own example scripts to this directory:

1. **Language-specific examples** - Python, JavaScript, Go, etc.
2. **Integration examples** - Specific tools/frameworks
3. **Advanced usage** - Batch processing, streaming, etc.

**Template for new examples:**

```python
#!/usr/bin/env python3
"""
Your Example Name

Brief description of what this example demonstrates.

Usage:
    python examples/your_example.py
"""

import os
import requests

# Get configuration from environment
BASE_URL = os.getenv("PROXY_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("PROXY_API_KEY")

if not API_KEY:
    print("Error: PROXY_API_KEY not set")
    exit(1)

# Your example code here
...
```

## Running Examples

**Prerequisites:**
1. Proxy must be running: `python src/proxy_app/main.py`
2. Set environment variables:
   ```bash
   export PROXY_API_KEY="your-proxy-api-key"
   export PROXY_BASE_URL="http://localhost:8000"  # Optional
   ```

**Run any example:**
```bash
python examples/example_name.py
```

## Troubleshooting

**"Connection refused"**
- Ensure proxy is running on the specified URL
- Check `--host` and `--port` settings

**"401 Unauthorized"**
- Verify PROXY_API_KEY matches your `.env` file
- Check the API key in your command/environment

**"No models available"**
- Add provider API keys to `.env`
- Run: `python src/proxy_app/main.py --add-credential`

**Test failures**
- Enable detailed logging: `--enable-request-logging`
- Check logs in `logs/` directory
- Verify provider API keys are valid

## Contributing

To contribute a new example:

1. Create your example file in this directory
2. Follow the template structure above
3. Add documentation to this README
4. Test thoroughly with different configurations
5. Submit a pull request

## Additional Resources

- [Main README](../README.md) - Project overview
- [Claude Code Setup Guide](../CLAUDE_CODE_SETUP.md) - Claude Code integration
- [API Guide](../ANTHROPIC_API_GUIDE.md) - Anthropic API compatibility
- [Technical Documentation](../DOCUMENTATION.md) - Architecture details

---

**Need help?** Open an issue on [GitHub](https://github.com/Mirrowel/LLM-API-Key-Proxy/issues)
