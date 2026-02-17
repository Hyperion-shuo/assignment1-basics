"""
Load trained BPE tokenizer and compute compression ratios for questions a and b.

(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories
    and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these
    sampled documents into integer IDs. What is each tokenizer's compression ratio (bytes/token)?

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare
    the compression ratio and/or qualitatively describe what happens.
"""

import os
import sys
import json
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import TrainedTokenizer


def load_tokenizer_from_files(vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> TrainedTokenizer:
    """
    Load a trained tokenizer from vocab and merges files.
    
    Args:
        vocab_filepath: Path to the vocab JSON file 
                       (format: {token_id: [byte1, byte2, ...], ...})
        merges_filepath: Path to the merges JSON file 
                        (format: [[[byte1, byte2, ...], [byte3, byte4, ...]], ...])
        special_tokens: Optional list of special tokens
    
    Returns:
        TrainedTokenizer instance
    """
    # Load vocab: {token_id: [byte1, byte2, ...]}
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        vocab_bytes_list = json.load(f)
    
    # Convert to {int: bytes}
    vocab = {}
    for token_id_str, byte_list in vocab_bytes_list.items():
        token_id = int(token_id_str)
        token_bytes = bytes(byte_list)
        vocab[token_id] = token_bytes
    
    # Load merges from JSON: [[[byte1, ...], [byte2, ...]], ...]
    with open(merges_filepath, 'r', encoding='utf-8') as f:
        merges_list = json.load(f)
    
    # Convert to list of (bytes, bytes) tuples
    merges = []
    for merge_pair in merges_list:
        token1_bytes = bytes(merge_pair[0])
        token2_bytes = bytes(merge_pair[1])
        merges.append((token1_bytes, token2_bytes))
    
    return TrainedTokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def sample_documents(filepath: str, num_samples: int = 10, delimiter: str = "<|endoftext|>") -> list[str]:
    """
    Sample documents from a text file separated by delimiter.
    
    Args:
        filepath: Path to the text file
        num_samples: Number of documents to sample
        delimiter: Document delimiter
    
    Returns:
        List of sampled document strings
    """
    print(f"Reading documents from {filepath}...")
    documents = []
    current_doc = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if delimiter in line:
                # Split by delimiter
                parts = line.split(delimiter)
                for i, part in enumerate(parts):
                    if i > 0:
                        # Finish current document
                        if current_doc:
                            documents.append(''.join(current_doc))
                            current_doc = []
                    if part.strip():
                        current_doc.append(part)
            else:
                current_doc.append(line)
        
        # Add last document
        if current_doc:
            documents.append(''.join(current_doc))
    
    print(f"Total documents: {len(documents)}")
    
    # Sample random documents
    if len(documents) <= num_samples:
        return documents
    
    random.seed(42)  # For reproducibility
    sampled = random.sample(documents, num_samples)
    return sampled


def compute_compression_ratio(tokenizer: TrainedTokenizer, documents: list[str]) -> float:
    """
    Compute compression ratio (bytes/token) for a list of documents.
    
    Args:
        tokenizer: The tokenizer to use
        documents: List of document strings
    
    Returns:
        Compression ratio (bytes/token)
    """
    total_bytes = 0
    total_tokens = 0
    
    for doc in documents:
        doc_bytes = len(doc.encode('utf-8'))
        tokens = tokenizer.encode(doc)
        total_bytes += doc_bytes
        total_tokens += len(tokens)
    
    return total_bytes / total_tokens if total_tokens > 0 else 0


def main():
    # Data paths
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # TinyStories paths (10K vocab)
    tinystories_vocab = os.path.join(data_dir, 'tinystories_bpe_vocab.json')
    tinystories_merges = os.path.join(data_dir, 'tinystories_bpe_merges.json')
    tinystories_train = os.path.join(data_dir, 'TinyStoriesV2-GPT4-train.txt')
    
    # OpenWebText paths (32K vocab)
    owt_vocab = os.path.join(data_dir, 'owt_train_bpe_vocab.json')
    owt_merges = os.path.join(data_dir, 'owt_train_bpe_merges.json')
    owt_train = os.path.join(data_dir, 'owt_train.txt')
    
    # Load tokenizers
    print("=" * 60)
    print("Loading tokenizers...")
    print("=" * 60)
    
    print("Loading TinyStories tokenizer (10K vocab)...")
    tinystories_tokenizer = load_tokenizer_from_files(tinystories_vocab, tinystories_merges)
    print(f"TinyStories vocab size: {len(tinystories_tokenizer.vocab)}")
    
    print("\nLoading OpenWebText tokenizer (32K vocab)...")
    owt_tokenizer = load_tokenizer_from_files(owt_vocab, owt_merges)
    print(f"OpenWebText vocab size: {len(owt_tokenizer.vocab)}")
    
    # =========================================================
    # Question (a): Sample 10 documents and compute compression ratios
    # =========================================================
    print("\n" + "=" * 60)
    print("Question (a): Compression ratios with matching tokenizers")
    print("=" * 60)
    
    # Sample documents
    print("\nSampling 10 documents from TinyStories...")
    tinystories_docs = sample_documents(tinystories_train, num_samples=10)
    
    print("\nSampling 10 documents from OpenWebText...")
    owt_docs = sample_documents(owt_train, num_samples=10)
    
    # Compute compression ratios
    print("\nComputing compression ratios...")
    
    # TinyStories with TinyStories tokenizer
    ts_compression = compute_compression_ratio(tinystories_tokenizer, tinystories_docs)
    print(f"\nTinyStories tokenizer on TinyStories: {ts_compression:.4f} bytes/token")
    
    # OpenWebText with OpenWebText tokenizer
    owt_compression = compute_compression_ratio(owt_tokenizer, owt_docs)
    print(f"OpenWebText tokenizer on OpenWebText: {owt_compression:.4f} bytes/token")
    
    # =========================================================
    # Question (b): Cross-tokenization - TinyStories tokenizer on OpenWebText
    # =========================================================
    print("\n" + "=" * 60)
    print("Question (b): TinyStories tokenizer on OpenWebText samples")
    print("=" * 60)
    
    # TinyStories tokenizer on OpenWebText
    cross_compression = compute_compression_ratio(tinystories_tokenizer, owt_docs)
    print(f"\nTinyStories tokenizer on OpenWebText: {cross_compression:.4f} bytes/token")
    
    # Compare with native tokenizer
    print(f"\nComparison:")
    print(f"  - OpenWebText tokenizer on OpenWebText: {owt_compression:.4f} bytes/token")
    print(f"  - TinyStories tokenizer on OpenWebText: {cross_compression:.4f} bytes/token")
    print(f"  - Difference: {cross_compression - owt_compression:.4f} bytes/token")
    print(f"  - Ratio: {cross_compression / owt_compression:.2f}x")
    
    # Show example encoding
    print("\n" + "=" * 60)
    print("Example encoding comparison:")
    print("=" * 60)
    
    sample_text = owt_docs[0][:500] if owt_docs else "Sample text"
    print(f"\nSample text (first 500 chars):\n{sample_text[:200]}...")
    
    owt_tokens = owt_tokenizer.encode(sample_text)
    ts_tokens = tinystories_tokenizer.encode(sample_text)
    
    print(f"\nOpenWebText tokenizer: {len(owt_tokens)} tokens")
    print(f"TinyStories tokenizer: {len(ts_tokens)} tokens")
    print(f"Token ratio: {len(ts_tokens) / len(owt_tokens):.2f}x")

if __name__ == "__main__":
    main()