#!/usr/bin/env python3
"""
Anthropic API Compatibility Test Script

This script tests the proxy's Anthropic Messages API compatibility.
It validates:
- Basic non-streaming requests
- Streaming responses with SSE
- Tool calling (function calling)
- Error handling
- Authentication

Usage:
    python examples/anthropic_test.py
    
    Or with custom settings:
    python examples/anthropic_test.py --base-url http://localhost:8000 --api-key YOUR_KEY
"""

import argparse
import json
import sys
import time
from typing import Optional, Dict, Any
import requests


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class AnthropicProxyTester:
    """Test suite for Anthropic API compatibility"""
    
    def __init__(self, base_url: str, api_key: str, model: str = "openai/gpt-4o-mini"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json"
        }
        self.tests_passed = 0
        self.tests_failed = 0
        
    def print_header(self, text: str):
        """Print a test section header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
        
    def print_success(self, text: str):
        """Print success message"""
        print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")
        self.tests_passed += 1
        
    def print_error(self, text: str):
        """Print error message"""
        print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")
        self.tests_failed += 1
        
    def print_info(self, text: str):
        """Print info message"""
        print(f"{Colors.OKCYAN}{text}{Colors.ENDC}")
        
    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")
        
    def test_health_check(self) -> bool:
        """Test 1: Basic health check"""
        self.print_header("Test 1: Health Check")
        
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("Status") == "API Key Proxy is running":
                    self.print_success("Proxy is running")
                    return True
                else:
                    self.print_error(f"Unexpected response: {data}")
                    return False
            else:
                self.print_error(f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.print_error(f"Health check failed: {e}")
            return False
    
    def test_list_models(self) -> bool:
        """Test 2: List available models"""
        self.print_header("Test 2: List Models")
        
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                
                if len(models) > 0:
                    self.print_success(f"Found {len(models)} available models")
                    self.print_info(f"First 5 models: {[m['id'] for m in models[:5]]}")
                    
                    # Check if our test model is available
                    model_ids = [m['id'] for m in models]
                    if any(self.model in mid for mid in model_ids):
                        self.print_success(f"Test model '{self.model}' is available")
                    else:
                        self.print_warning(f"Test model '{self.model}' not found. Using first available.")
                        if models:
                            self.model = models[0]['id']
                            self.print_info(f"Switched to: {self.model}")
                    
                    return True
                else:
                    self.print_error("No models available")
                    return False
            else:
                self.print_error(f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.print_error(f"List models failed: {e}")
            return False
    
    def test_basic_message(self) -> bool:
        """Test 3: Basic non-streaming message"""
        self.print_header("Test 3: Basic Message (Non-Streaming)")
        
        try:
            payload = {
                "model": self.model,
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'Hello, World!' and nothing else."
                    }
                ]
            }
            
            self.print_info(f"Sending request to {self.base_url}/v1/messages")
            self.print_info(f"Model: {self.model}")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                required_fields = ["id", "type", "role", "content", "model", "stop_reason", "usage"]
                missing_fields = [f for f in required_fields if f not in data]
                
                if missing_fields:
                    self.print_error(f"Missing fields in response: {missing_fields}")
                    return False
                
                # Validate values
                if data["type"] != "message":
                    self.print_error(f"Expected type 'message', got '{data['type']}'")
                    return False
                
                if data["role"] != "assistant":
                    self.print_error(f"Expected role 'assistant', got '{data['role']}'")
                    return False
                
                if not isinstance(data["content"], list) or len(data["content"]) == 0:
                    self.print_error("Content is not a non-empty array")
                    return False
                
                content_block = data["content"][0]
                if content_block.get("type") != "text":
                    self.print_error(f"Expected content type 'text', got '{content_block.get('type')}'")
                    return False
                
                response_text = content_block.get("text", "")
                self.print_success("Response structure is valid")
                self.print_info(f"Response: {response_text[:100]}...")
                self.print_info(f"Tokens used: {data['usage']['input_tokens']} in, {data['usage']['output_tokens']} out")
                
                return True
            else:
                self.print_error(f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.print_error(f"Basic message test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_streaming_message(self) -> bool:
        """Test 4: Streaming message with SSE"""
        self.print_header("Test 4: Streaming Message (SSE)")
        
        try:
            payload = {
                "model": self.model,
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from 1 to 3, one number per line."
                    }
                ],
                "stream": True
            }
            
            self.print_info("Sending streaming request...")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                self.print_error(f"HTTP {response.status_code}: {response.text}")
                return False
            
            # Track events
            events_received = {
                "message_start": 0,
                "content_block_start": 0,
                "content_block_delta": 0,
                "content_block_stop": 0,
                "message_delta": 0,
                "message_stop": 0
            }
            
            accumulated_text = ""
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                line = line.decode('utf-8')
                
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                    if event_type in events_received:
                        events_received[event_type] += 1
                
                elif line.startswith("data: "):
                    data_str = line[6:].strip()
                    try:
                        data = json.loads(data_str)
                        
                        # Extract text from content_block_delta events
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                accumulated_text += text
                    except json.JSONDecodeError:
                        pass
            
            # Validate events
            self.print_info(f"Events received: {events_received}")
            self.print_info(f"Accumulated text: {accumulated_text}")
            
            if events_received["message_start"] != 1:
                self.print_error("Expected exactly 1 message_start event")
                return False
            
            if events_received["message_stop"] != 1:
                self.print_error("Expected exactly 1 message_stop event")
                return False
            
            if events_received["content_block_delta"] == 0:
                self.print_error("Expected at least 1 content_block_delta event")
                return False
            
            if not accumulated_text:
                self.print_error("No text content received")
                return False
            
            self.print_success("Streaming response is valid")
            self.print_success(f"Received {sum(events_received.values())} total events")
            
            return True
            
        except Exception as e:
            self.print_error(f"Streaming test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_tool_calling(self) -> bool:
        """Test 5: Tool calling (function calling)"""
        self.print_header("Test 5: Tool Calling")
        
        # Skip tool calling test for models that don't support it well
        if "gpt-3.5" in self.model.lower() or "flash" in self.model.lower():
            self.print_warning("Skipping tool calling test for this model (may not support it)")
            return True
        
        try:
            payload = {
                "model": self.model,
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
                                }
                            },
                            "required": ["location"]
                        }
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Paris, France?"
                    }
                ]
            }
            
            self.print_info("Sending tool calling request...")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if model used tool
                content = data.get("content", [])
                tool_uses = [block for block in content if block.get("type") == "tool_use"]
                
                if tool_uses:
                    self.print_success(f"Model used {len(tool_uses)} tool(s)")
                    
                    for tool_use in tool_uses:
                        tool_name = tool_use.get("name")
                        tool_input = tool_use.get("input", {})
                        self.print_info(f"Tool: {tool_name}")
                        self.print_info(f"Input: {json.dumps(tool_input, indent=2)}")
                    
                    if data.get("stop_reason") == "tool_use":
                        self.print_success("Stop reason is 'tool_use' (correct)")
                    else:
                        self.print_warning(f"Stop reason is '{data.get('stop_reason')}' (expected 'tool_use')")
                    
                    return True
                else:
                    self.print_warning("Model didn't use tools (generated text response instead)")
                    self.print_info("This is acceptable - model chose not to use tools")
                    return True
            else:
                self.print_error(f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.print_error(f"Tool calling test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_authentication(self) -> bool:
        """Test 6: Authentication validation"""
        self.print_header("Test 6: Authentication")
        
        try:
            # Test with invalid key
            self.print_info("Testing with invalid API key...")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": "invalid-key-12345",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=10
            )
            
            if response.status_code == 401:
                self.print_success("Invalid key correctly rejected (401)")
            else:
                self.print_error(f"Expected 401, got {response.status_code}")
                return False
            
            # Test with missing key
            self.print_info("Testing with missing API key...")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=10
            )
            
            if response.status_code == 401:
                self.print_success("Missing key correctly rejected (401)")
            else:
                self.print_error(f"Expected 401, got {response.status_code}")
                return False
            
            # Test with valid key (using Bearer format)
            self.print_info("Testing with valid key (Bearer format)...")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                self.print_success("Valid key (Bearer format) accepted")
            else:
                self.print_error(f"Expected 200, got {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            self.print_error(f"Authentication test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test 7: Error handling"""
        self.print_header("Test 7: Error Handling")
        
        try:
            # Test missing required field (max_tokens)
            self.print_info("Testing missing max_tokens...")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "test"}]
                },
                timeout=10
            )
            
            # Should return 400 or succeed (max_tokens might be optional in some implementations)
            if response.status_code == 400:
                self.print_success("Missing max_tokens correctly rejected")
            elif response.status_code == 200:
                self.print_info("max_tokens treated as optional (acceptable)")
            else:
                self.print_warning(f"Unexpected status code: {response.status_code}")
            
            # Test invalid JSON
            self.print_info("Testing invalid JSON...")
            
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                data="invalid json{{{",
                timeout=10
            )
            
            if response.status_code == 400:
                self.print_success("Invalid JSON correctly rejected")
            else:
                self.print_error(f"Expected 400, got {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            self.print_error(f"Error handling test failed: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests and print summary"""
        print(f"\n{Colors.BOLD}{Colors.HEADER}")
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║        Anthropic API Compatibility Test Suite                    ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        
        print(f"{Colors.OKBLUE}Base URL: {self.base_url}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Model: {self.model}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}API Key: {self.api_key[:8]}...{Colors.ENDC}")
        
        # Run tests
        tests = [
            ("Health Check", self.test_health_check),
            ("List Models", self.test_list_models),
            ("Basic Message", self.test_basic_message),
            ("Streaming", self.test_streaming_message),
            ("Tool Calling", self.test_tool_calling),
            ("Authentication", self.test_authentication),
            ("Error Handling", self.test_error_handling),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Tests interrupted by user{Colors.ENDC}")
                break
            except Exception as e:
                self.print_error(f"Unexpected error in {test_name}: {e}")
        
        # Print summary
        self.print_header("Test Summary")
        
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"{Colors.OKGREEN}Passed: {self.tests_passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {self.tests_failed}{Colors.ENDC}")
        print(f"{Colors.BOLD}Total: {total_tests}{Colors.ENDC}")
        print(f"{Colors.BOLD}Pass Rate: {pass_rate:.1f}%{Colors.ENDC}")
        
        if self.tests_failed == 0:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ All tests passed!{Colors.ENDC}")
            return True
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}✗ Some tests failed{Colors.ENDC}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Anthropic API compatibility of the proxy"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Proxy base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        required=False,
        help="Proxy API key (PROXY_API_KEY from .env)"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model to test with (default: openai/gpt-4o-mini)"
    )
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key
    if not api_key:
        import os
        api_key = os.getenv("PROXY_API_KEY")
    
    if not api_key:
        print(f"{Colors.FAIL}Error: API key not provided and PROXY_API_KEY not found in environment{Colors.ENDC}")
        print(f"{Colors.INFO}Usage: python anthropic_test.py --api-key YOUR_KEY{Colors.ENDC}")
        print(f"{Colors.INFO}Or set PROXY_API_KEY environment variable{Colors.ENDC}")
        sys.exit(1)
    
    # Run tests
    tester = AnthropicProxyTester(args.base_url, api_key, args.model)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
