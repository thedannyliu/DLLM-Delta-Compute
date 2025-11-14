#!/usr/bin/env python3
"""
Calculate GSM8K accuracy from evaluation logs by counting correct predictions.
"""

import re
import json
import sys
from pathlib import Path

def calculate_gsm8k_accuracy(log_file):
    """
    Count correct vs total for GSM8K.
    We count responses that produce a valid numerical answer.
    """
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Count total test questions (those with empty Answer: followed by Response:)
    # Pattern: Question: ... Answer:\nResponse:\n...
    test_pattern = r'Question: (.*?)\nAnswer:\s*\nResponse:\s*\n(.*?)(?=\n(?:Question:|Context:|diffllm|Running))'
    test_cases = re.findall(test_pattern, content, re.DOTALL)
    
    print(f"Found {len(test_cases)} test cases")
    
    if len(test_cases) == 0:
        return None
    
    # For each test case, check if response contains "#### <number>"
    valid_count = 0
    for i, (question, response) in enumerate(test_cases):
        # Check if response has the answer format
        answer_match = re.search(r'#### (\d+)', response)
        if answer_match:
            valid_count += 1
            answer = answer_match.group(1)
            print(f"  Test {i+1}: Valid answer (#### {answer})")
        else:
            print(f"  Test {i+1}: Invalid/malformed response")
    
    # Extract timing
    timing_match = re.search(r'100/100 \[(.+?)<', content)
    if timing_match:
        total_time_str = timing_match.group(1)
        print(f"\nTotal evaluation time: {total_time_str}")
    
    return {
        'total_test_cases': len(test_cases),
        'valid_responses': valid_count,
        'invalid_responses': len(test_cases) - valid_count,
        'valid_response_rate': valid_count / len(test_cases) if len(test_cases) > 0 else 0,
        'note': 'Without ground truth labels, showing valid response count only'
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python count_gsm8k_responses.py <log_file>")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: {log_file} not found")
        sys.exit(1)
    
    print(f"Analyzing {log_file}...\n")
    
    results = calculate_gsm8k_accuracy(log_file)
    
    if results:
        print(f"\n{'='*60}")
        print(f"Total test cases: {results['total_test_cases']}")
        print(f"Valid responses: {results['valid_responses']}")
        print(f"Invalid/malformed: {results['invalid_responses']}")
        print(f"Valid response rate: {results['valid_response_rate']:.1%}")
        print(f"{'='*60}")
        
        if len(sys.argv) > 2:
            output_file = Path(sys.argv[2])
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
    else:
        print("Failed to extract results")

if __name__ == '__main__':
    main()
