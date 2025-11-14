#!/usr/bin/env python3
"""
Extract evaluation results from SLURM log files.
Calculates accuracy, latency, and other metrics from lm-eval output.
"""

import re
import json
import sys
from pathlib import Path
from datetime import datetime

def extract_gsm8k_accuracy(log_file):
    """Extract GSM8K accuracy by comparing responses to ground truth."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # GSM8K format: Question, then Answer (empty for test), then Response with prediction
    # Ground truth is stored separately in the dataset
    # We need to look for the lm-eval metrics output instead
    
    # Try to find accuracy in lm-eval output
    accuracy_match = re.search(r'exact_match.*?:\s*([0-9.]+)', content)
    if accuracy_match:
        accuracy = float(accuracy_match.group(1))
        # Try to find n parameter
        n_match = re.search(r'"n":\s*(\d+)', content)
        total = int(n_match.group(1)) if n_match else 100
        correct = int(accuracy * total)
        return {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }
    
    # Fallback: count responses manually
    # For GSM8K, the response should contain "#### <number>"
    response_pattern = r'Question:.*?Answer:\s*\nResponse:\s*\n(.*?)(?=\n(?:Question:|diffllm))'
    responses = re.findall(response_pattern, content, re.DOTALL)
    
    if not responses:
        print(f"No responses found in {log_file}")
        return None
    
    # Count responses with valid answers (#### format)
    answer_pattern = r'#### \d+'
    valid_responses = [r for r in responses if re.search(answer_pattern, r)]
    
    total = len(responses)
    # We can't verify correctness without ground truth, but we can count valid responses
    return {
        'valid_responses': len(valid_responses),
        'total_responses': total,
        'note': 'Accuracy unavailable - need ground truth labels or lm-eval metrics'
    }

def extract_timing(log_file):
    """Extract timing information from log."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find start and end times
    start_match = re.search(r'Started at: (.+)', content)
    end_match = re.search(r'Evaluation completed at (.+)', content)
    
    if not start_match or not end_match:
        return None
    
    start_str = start_match.group(1).strip()
    end_str = end_match.group(1).strip()
    
    # Parse times
    fmt = "%a %b %d %H:%M:%S %Z %Y"
    try:
        start_time = datetime.strptime(start_str, fmt)
        end_time = datetime.strptime(end_str, fmt)
        duration = (end_time - start_time).total_seconds()
        return {
            'start': start_str,
            'end': end_str,
            'duration_seconds': duration,
            'duration_minutes': duration / 60
        }
    except:
        return None

def extract_resource_usage(log_file):
    """Extract resource usage from SLURM epilog."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find resource usage line
    rsrc_match = re.search(r'Rsrc Used:\s+(.+)', content)
    if not rsrc_match:
        return None
    
    rsrc_str = rsrc_match.group(1)
    
    # Parse individual metrics
    metrics = {}
    for item in rsrc_str.split(','):
        if '=' in item:
            key, value = item.strip().split('=', 1)
            metrics[key] = value
    
    return metrics

def extract_model_config(log_file):
    """Extract model configuration from log."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find model args
    config_match = re.search(r'diffllm \((.+?)\), gen_kwargs:', content)
    if not config_match:
        return None
    
    config_str = config_match.group(1)
    config = {}
    for item in config_str.split(','):
        if '=' in item:
            key, value = item.strip().split('=', 1)
            config[key] = value
    
    return config

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_results.py <log_file> [output_json]")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: {log_file} not found")
        sys.exit(1)
    
    print(f"Extracting results from {log_file}...")
    
    results = {
        'log_file': str(log_file),
        'extracted_at': datetime.now().isoformat()
    }
    
    # Extract all metrics
    accuracy = extract_gsm8k_accuracy(log_file)
    timing = extract_timing(log_file)
    resources = extract_resource_usage(log_file)
    config = extract_model_config(log_file)
    
    if accuracy:
        results['accuracy'] = accuracy
        print(f"✅ Accuracy: {accuracy['correct']}/{accuracy['total']} = {accuracy['accuracy']:.2%}")
    
    if timing:
        results['timing'] = timing
        print(f"✅ Duration: {timing['duration_minutes']:.2f} minutes")
    
    if resources:
        results['resources'] = resources
        print(f"✅ Memory used: {resources.get('mem', 'N/A')}")
    
    if config:
        results['config'] = config
        print(f"✅ Model: {config.get('pretrained', 'N/A')}")
    
    # Save to JSON if output file specified
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 Results saved to {output_file}")
    else:
        print("\n📊 Full results:")
        print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
