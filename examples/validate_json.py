#!/usr/bin/env python3
"""
Validate JSON examples in documentation files

This script extracts and validates all JSON examples from documentation.
"""

import json
import re
import sys
from pathlib import Path

def extract_json_blocks(content):
    """Extract JSON code blocks from markdown"""
    # Match JSON blocks in ```json or ```json\n format
    pattern = r'```json\s*\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    # Also match plain code blocks that look like JSON
    pattern2 = r'```\s*\n(\{.*?\})\s*```'
    matches2 = re.findall(pattern2, content, re.DOTALL)
    
    return matches + matches2

def validate_json(json_str, location):
    """Validate a JSON string"""
    try:
        json.loads(json_str)
        return True, None
    except json.JSONDecodeError as e:
        return False, f"{location}: {e}"

def main():
    docs_dir = Path(__file__).parent.parent
    
    # Files to check
    files_to_check = [
        'CLAUDE_CODE_SETUP.md',
        'ANTHROPIC_API_GUIDE.md',
        'examples/README.md',
        '.env.example',
    ]
    
    total_checks = 0
    passed = 0
    failed = 0
    errors = []
    
    print("🔍 Validating JSON examples in documentation...\n")
    
    for filename in files_to_check:
        filepath = docs_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  {filename} not found, skipping")
            continue
        
        print(f"📄 Checking {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        json_blocks = extract_json_blocks(content)
        
        if not json_blocks:
            print(f"   No JSON blocks found\n")
            continue
        
        for i, json_block in enumerate(json_blocks):
            total_checks += 1
            location = f"{filename}:block{i+1}"
            
            valid, error = validate_json(json_block, location)
            
            if valid:
                passed += 1
                print(f"   ✓ Block {i+1}: Valid")
            else:
                failed += 1
                print(f"   ✗ Block {i+1}: Invalid - {error}")
                errors.append(error)
        
        print()
    
    # Summary
    print("━" * 70)
    print("📊 Validation Summary")
    print("━" * 70)
    print(f"Total JSON blocks: {total_checks}")
    print(f"✓ Valid: {passed}")
    print(f"✗ Invalid: {failed}")
    print("━" * 70)
    
    if failed > 0:
        print("\n❌ Errors found:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    else:
        print("\n✅ All JSON examples are valid!")
        sys.exit(0)

if __name__ == "__main__":
    main()
