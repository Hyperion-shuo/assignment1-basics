#!/usr/bin/env python3
"""
脚本：查找BPE词汇表中最长的token
"""

import json

def find_longest_token(vocab_file):
    """找出词汇表中最长的token"""
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    longest_token = ""
    longest_length = 0
    longest_id = None
    
    for token_id, token in vocab.items():
        token_length = len(token)
        if token_length > longest_length:
            longest_length = token_length
            longest_token = token
            longest_id = token_id
    
    return longest_id, longest_token, longest_length

if __name__ == "__main__":
    # tinystories_bpe_vocab.json
    vocab_file = "data/owt_train_bpe_vocab.json"
    
    token_id, token, length = find_longest_token(vocab_file)
    
    print(f"最长的token:")
    print(f"  ID: {token_id}")
    print(f"  内容: {repr(token)}")
    print(f"  长度: {length} 个字符")
