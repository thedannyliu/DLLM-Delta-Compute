#!/bin/bash
# Verify Dream baseline can run

cd "$(dirname "$0")/../external/Dream"

echo "=== Verifying Dream Model Loading ==="

python3 -c "
import torch
from transformers import AutoModel, AutoTokenizer

print('Loading Dream model...')
model_path = 'Dream-org/Dream-v0-Instruct-7B'

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print('✓ Tokenizer loaded')
    
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
        device_map='auto'
    )
    print('✓ Model loaded')
    
    # Test generation
    messages = [{'role': 'user', 'content': '2+2=?'}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors='pt', return_dict=True, add_generation_prompt=True
    )
    
    print('✓ Tokenization works')
    print(f'Model device: {next(model.parameters()).device}')
    print(f'Model dtype: {next(model.parameters()).dtype}')
    print('✓ Dream baseline verification passed!')
    
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
"
