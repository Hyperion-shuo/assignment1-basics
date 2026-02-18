"""
Tokenize OWT data with streaming approach for low memory usage.
Uses find_chunk_boundaries for efficient file chunking.

Usage:
    # Test mode (sample 64 documents, verify roundtrip)
    uv run python my_scripts/tokenize_owt_data.py --test
    
    # Full mode (stream process entire files)
    uv run python my_scripts/tokenize_owt_data.py
"""
import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from cs336_basics.tokenizer import TrainedTokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
VOCAB_PATH = os.path.join(DATA_DIR, 'owt_train_bpe_vocab.json')
MERGES_PATH = os.path.join(DATA_DIR, 'owt_train_bpe_merges.json')
TRAIN_PATH = os.path.join(DATA_DIR, 'owt_train.txt')
VAL_PATH = os.path.join(DATA_DIR, 'owt_valid.txt')
SPECIAL_TOKENS = ["<|endoftext|>"]


def tokenize_file_streaming(tokenizer: TrainedTokenizer, filepath: str, 
                            num_chunks: int = 32) -> list[int]:
    """
    Stream-tokenize a file using find_chunk_boundaries.
    Memory efficient: only one chunk in memory at a time.
    """
    delimiter = b"<|endoftext|>"
    all_tokens = []
    
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, delimiter)
        
        for i in tqdm(range(len(boundaries) - 1), desc=f"Processing {os.path.basename(filepath)}"):
            start, end = boundaries[i], boundaries[i + 1]
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            
            if chunk_text:
                tokens = tokenizer.encode(chunk_text)
                all_tokens.extend(tokens)
    
    return all_tokens


def tokenize_and_save_streaming(tokenizer: TrainedTokenizer, filepath: str, 
                                output_path: str, num_chunks: int = 32,
                                chunk_size_mb: int = 200):
    """
    Stream-tokenize a file and save to multiple npz files.
    Each npz file is ~chunk_size_mb MB.
    """
    delimiter = b"<|endoftext|>"
    buffer = []
    buffer_tokens = 0
    file_idx = 0
    total_tokens = 0
    tokens_per_mb = chunk_size_mb * 1024 * 1024 // 2  # uint16 = 2 bytes
    
    base_name = output_path.replace('.npy', '')
    
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, delimiter)
        
        for i in tqdm(range(len(boundaries) - 1), desc=f"Processing {os.path.basename(filepath)}"):
            start, end = boundaries[i], boundaries[i + 1]
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            
            if chunk_text:
                tokens = tokenizer.encode(chunk_text)
                buffer.extend(tokens)
                buffer_tokens += len(tokens)
                total_tokens += len(tokens)
                
                # Save when buffer is large enough (loop to handle large chunks)
                while buffer_tokens >= tokens_per_mb:
                    # Take tokens_per_mb tokens to save
                    save_tokens = buffer[:tokens_per_mb]
                    buffer = buffer[tokens_per_mb:]
                    buffer_tokens = len(buffer)
                    
                    arr = np.array(save_tokens, dtype=np.uint16)
                    np.savez_compressed(f"{base_name}_{file_idx}.npz", tokens=arr)
                    print(f"  Saved {base_name}_{file_idx}.npz ({len(save_tokens)} tokens)")
                    file_idx += 1
    
    # Save remaining tokens
    if buffer:
        if file_idx == 0:
            # Only one file, save as .npy
            arr = np.array(buffer, dtype=np.uint16)
            np.save(output_path, arr)
            print(f"  Saved {output_path} ({len(buffer)} tokens)")
        else:
            arr = np.array(buffer, dtype=np.uint16)
            np.savez_compressed(f"{base_name}_{file_idx}.npz", tokens=arr)
            print(f"  Saved {base_name}_{file_idx}.npz ({len(buffer)} tokens)")
    
    return total_tokens


def test_mode(tokenizer: TrainedTokenizer, num_samples: int = 64):
    """Test encoding pipeline: encode -> save -> load -> decode roundtrip."""
    import tempfile
    import random
    
    print("\n" + "=" * 60)
    print("TEST MODE: Verifying roundtrip consistency")
    print("=" * 60)
    
    # Read a small portion of train file
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        text = f.read(1024 * 1024)  # Read 1MB
    
    # Split into documents and sample
    docs = text.split("<|endoftext|>")[:num_samples]
    test_text = "<|endoftext|>".join(docs) + "<|endoftext|>"
    
    print(f"\nTest text: {len(test_text)} chars, ~{num_samples} documents")
    
    # Encode
    start = time.time()
    tokens = tokenizer.encode(test_text)
    encode_time = time.time() - start
    print(f"Encode: {len(tokens)} tokens in {encode_time:.2f}s")
    
    # Save & Load
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        tmp_path = f.name
    
    arr = np.array(tokens, dtype=np.uint16)
    np.save(tmp_path, arr)
    loaded = np.load(tmp_path)
    os.unlink(tmp_path)
    
    # Decode
    start = time.time()
    decoded = tokenizer.decode(loaded.tolist())
    decode_time = time.time() - start
    print(f"Decode: {len(decoded)} chars in {decode_time:.2f}s")
    
    # Verify
    if decoded == test_text:
        print("\n✓ Roundtrip PASSED: original == decoded")
        return True
    else:
        print("\n✗ Roundtrip FAILED")
        print(f"  Original: {len(test_text)} chars")
        print(f"  Decoded:  {len(decoded)} chars")
        return False


def full_mode(tokenizer: TrainedTokenizer, num_chunks: int = 32, chunk_size_mb: int = 200):
    """Full tokenization mode with streaming and chunked saving."""
    print("\n" + "=" * 60)
    print("FULL MODE: Tokenizing OWT dataset")
    print(f"  - Processing chunks: {num_chunks}")
    print(f"  - Output file size: ~{chunk_size_mb}MB each")
    print("=" * 60)
    
    total_start = time.time()
    
    # Process train
    print("\n>>> Processing train file...")
    train_output = os.path.join(DATA_DIR, 'owt_train_tokens.npy')
    train_start = time.time()
    train_tokens = tokenize_and_save_streaming(
        tokenizer, TRAIN_PATH, train_output, num_chunks, chunk_size_mb
    )
    train_time = time.time() - train_start
    print(f"Train: {train_tokens} tokens total")
    print(f"Train time: {train_time:.2f}s")
    
    # Process val
    print("\n>>> Processing val file...")
    val_output = os.path.join(DATA_DIR, 'owt_val_tokens.npy')
    val_start = time.time()
    val_tokens = tokenize_and_save_streaming(
        tokenizer, VAL_PATH, val_output, 1, chunk_size_mb
    )
    val_time = time.time() - val_start
    print(f"Val: {val_tokens} tokens total")
    print(f"Val time: {val_time:.2f}s")
    
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Train: {train_tokens} tokens")
    print(f"  Val:   {val_tokens} tokens")
    print(f"  Train time: {train_time:.2f}s")
    print(f"  Val time:   {val_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")

# uv run python -m my_scripts.tokenize_owt_data --test
def main():
    parser = argparse.ArgumentParser(description='Tokenize OWT data')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--num-samples', type=int, default=64, help='Samples for test mode')
    parser.add_argument('--num-chunks', type=int, default=32, help='Number of chunks for streaming')
    parser.add_argument('--chunk-size-mb', type=int, default=200, help='Output file size in MB')
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = TrainedTokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)
    
    if args.test:
        test_mode(tokenizer, args.num_samples)
    else:
        full_mode(tokenizer, args.num_chunks, args.chunk_size_mb)


if __name__ == "__main__":
    main()