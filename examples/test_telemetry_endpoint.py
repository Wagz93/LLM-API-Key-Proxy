#!/usr/bin/env python3
"""
Simple test for the Claude Code telemetry endpoint.

This script verifies that the /api/event_logging/batch endpoint:
1. Accepts POST requests without authentication
2. Returns 200 OK
3. Returns proper JSON response

Usage:
    python examples/test_telemetry_endpoint.py
    
Or with custom URL:
    python examples/test_telemetry_endpoint.py --base-url http://localhost:8000
"""

import argparse
import json
import sys
import requests


def test_telemetry_endpoint(base_url: str) -> bool:
    """Test the telemetry endpoint."""
    endpoint = f"{base_url}/api/event_logging/batch"
    
    print(f"Testing telemetry endpoint: {endpoint}")
    print("=" * 70)
    
    # Test 1: Empty payload
    print("\n[Test 1] Empty payload...")
    try:
        response = requests.post(
            endpoint,
            json={},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ Status: 200 OK")
            data = response.json()
            if data.get("status") == "ok":
                print("✓ Response: {'status': 'ok'}")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
        else:
            print(f"✗ Expected 200, got {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    # Test 2: Payload with events
    print("\n[Test 2] Payload with events array...")
    try:
        payload = {
            "events": [
                {
                    "type": "session_start",
                    "timestamp": "2026-01-01T15:00:00Z",
                    "metadata": {"version": "1.0"}
                },
                {
                    "type": "request_sent",
                    "timestamp": "2026-01-01T15:00:05Z",
                    "metadata": {"model": "claude-3-opus"}
                }
            ]
        }
        
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ Status: 200 OK")
            data = response.json()
            if data.get("status") == "ok":
                print("✓ Response: {'status': 'ok'}")
            else:
                print(f"✗ Unexpected response: {data}")
                return False
        else:
            print(f"✗ Expected 200, got {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    # Test 3: No authentication required
    print("\n[Test 3] No authentication required...")
    try:
        response = requests.post(
            endpoint,
            json={"events": []},
            headers={"Content-Type": "application/json"},
            # Note: No Authorization or x-api-key header
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ Endpoint accepts requests without authentication")
        else:
            print(f"✗ Expected 200, got {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    # Test 4: Invalid JSON (should still return 200)
    print("\n[Test 4] Invalid JSON handling...")
    try:
        response = requests.post(
            endpoint,
            data="invalid json{{{",
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ Endpoint gracefully handles invalid JSON (returns 200)")
        else:
            print(f"✗ Expected 200, got {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test the Claude Code telemetry endpoint"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Proxy base URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    # Remove trailing slash if present
    base_url = args.base_url.rstrip('/')
    
    # First check if proxy is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code != 200:
            print(f"Error: Proxy not responding at {base_url}")
            print("Please ensure the proxy is running.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: Cannot connect to proxy at {base_url}")
        print(f"  {e}")
        print("\nPlease ensure the proxy is running:")
        print("  python src/proxy_app/main.py")
        sys.exit(1)
    
    # Run tests
    success = test_telemetry_endpoint(base_url)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
