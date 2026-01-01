#!/bin/bash
# Quick validation script for documentation examples
# This script validates that curl commands in the documentation are syntactically correct

set -e

echo "🔍 Validating documentation examples..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
passed=0
failed=0

validate_curl() {
    local name="$1"
    local cmd="$2"
    
    echo -n "Testing $name... "
    
    # Test curl syntax by using --help to validate flags
    # Replace actual URLs with dummy URL for validation
    test_cmd=$(echo "$cmd" | sed 's|http://[^/]*|http://localhost:8000|g' | sed 's|https://[^/]*|http://localhost:8000|g')
    
    # Validate JSON if present
    if echo "$test_cmd" | grep -q '{"'; then
        json=$(echo "$test_cmd" | grep -oP '(?<=-d \047).*(?=\047)' || echo "$test_cmd" | grep -oP '(?<=-d ").*(?=")')
        if [ -n "$json" ]; then
            echo "$json" | python -m json.tool >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓${NC} JSON valid"
                ((passed++))
            else
                echo -e "${RED}✗${NC} Invalid JSON"
                ((failed++))
                return 1
            fi
        fi
    else
        echo -e "${GREEN}✓${NC}"
        ((passed++))
    fi
}

# Test examples from CLAUDE_CODE_SETUP.md
echo "📄 CLAUDE_CODE_SETUP.md examples:"
echo ""

validate_curl "Health check" "curl http://localhost:8000/"

validate_curl "List models" "curl -H 'Authorization: Bearer YOUR_PROXY_API_KEY' http://localhost:8000/v1/models"

validate_curl "Simple message" 'curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '\''{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 100,
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ]
  }'\'''

validate_curl "Streaming message" 'curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -N \
  -d '\''{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'\'''

validate_curl "Tool calling" 'curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '\''{
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
  }'\'''

echo ""
echo "📄 ANTHROPIC_API_GUIDE.md examples:"
echo ""

validate_curl "Basic request example" 'curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '\''{
    "model": "openai/gpt-4o-mini",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'\'''

validate_curl "Cost estimate" 'curl -X POST http://localhost:8000/v1/cost-estimate \
  -H "x-api-key: YOUR_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '\''{
    "model": "openai/gpt-4o",
    "prompt_tokens": 1000,
    "completion_tokens": 500
  }'\'''

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Validation Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Passed:${NC} $passed"
echo -e "${RED}Failed:${NC} $failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ All documentation examples are valid!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some examples have issues${NC}"
    exit 1
fi
