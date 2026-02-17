import pathlib
import os
import time
import argparse
import tracemalloc
import numpy as np
from cs336_basics.tokenizer import TrainedTokenizer


def get_peak_memory_mb():
    """Get current and peak memory usage in MB using tracemalloc."""
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 / 1024, peak / 1024 / 1024


def print_memory_stats(stage_name):
    """Print memory statistics for a given stage."""
    current_mb, peak_mb = get_peak_memory_mb()
    print(f"  [Memory] {stage_name}: Current={current_mb:.2f} MB, Peak={peak_mb:.2f} MB")


def load_documents(filepath, delimiter="<|endoftext|>"):
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

    print(f"{filepath} total documents: {len(documents)}")

    return documents


# us relative path to ensure it works regardless of current working directory
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# TinyStories paths (10K vocab)
owt_vocab = os.path.join(data_dir, 'owt_train_bpe_vocab.json')
owt_merges = os.path.join(data_dir, 'owt_train_bpe_merges.json')
owt_train = os.path.join(data_dir, 'owt_train.txt')
owt_val = os.path.join(data_dir, 'owt_valid.txt')

special_tokens = ["<|endoftext|>"]


def tokenize_documents(tokenizer, documents, delimiter="<|endoftext|>"):
    """
    Tokenize all documents and concatenate them into a single sequence of tokens.
    Each document is separated by the delimiter token.
    """
    all_tokens = []
    delimiter_tokens = tokenizer.encode(delimiter)
    
    for i, doc in enumerate(documents):
        # Encode the document
        tokens = tokenizer.encode(doc)
        all_tokens.extend(tokens)
        # Add delimiter between documents
        all_tokens.extend(delimiter_tokens)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(documents)} documents")
    
    return all_tokens

def tokenize_documents_multiprocess(tokenizer, documents, delimiter="<|endoftext|>", num_process=4):
    """
    Multi-process version of tokenize_documents for large document lists.
    
    Concatenates all documents with the delimiter and uses multi-process encoding.
    """
    # Join all documents with delimiter into a single string
    # Add delimiter at the end of each document
    text = delimiter.join(documents) + delimiter
    
    # encode_multiprocess expects a string and returns a flat list[int]
    all_tokens = tokenizer.encode_multiprocess(text, num_process=num_process)
    
    return all_tokens


def test_encoding_pipeline(tokenizer, documents, num_samples=10, num_process=4):
    """
    Test encoding pipeline: encode -> save -> load -> decode
    Tests both speed and correctness with a small sample of documents.
    """
    import tempfile
    import random
    
    print(f"\n{'='*60}")
    print(f"Testing encoding pipeline with {num_samples} sampled documents")
    print(f"{'='*60}")
    
    # Sample documents
    if len(documents) > num_samples:
        sampled_docs = random.sample(documents, num_samples)
    else:
        sampled_docs = documents[:num_samples]
    
    # Store original text for comparison
    delimiter = "<|endoftext|>"
    original_text = delimiter.join(sampled_docs) + delimiter
    
    print(f"\nOriginal text length: {len(original_text)} characters")
    print(f"Number of documents: {len(sampled_docs)}")
    
    # ========== ENCODE ==========
    print(f"\n--- Encoding ---")
    
    # Single-process encode
    single_start = time.time()
    tokens_single = tokenizer.encode(original_text)
    single_elapsed = time.time() - single_start
    print(f"Single-process encode: {len(tokens_single)} tokens in {single_elapsed:.4f}s")
    
    # Multi-process encode
    multi_start = time.time()
    tokens_multi = tokenizer.encode_multiprocess(original_text, num_process=num_process)
    multi_elapsed = time.time() - multi_start
    print(f"Multi-process encode ({num_process} processes): {len(tokens_multi)} tokens in {multi_elapsed:.4f}s")
    
    # Check encoding consistency
    if tokens_single == tokens_multi:
        print(f"✓ Encoding consistency: PASSED (single == multi)")
    else:
        print(f"✗ Encoding consistency: FAILED")
        print(f"  Single: {len(tokens_single)} tokens")
        print(f"  Multi:  {len(tokens_multi)} tokens")
        # Find first difference
        for i, (s, m) in enumerate(zip(tokens_single, tokens_multi)):
            if s != m:
                print(f"  First diff at position {i}: single={s}, multi={m}")
                break
    
    # ========== SAVE ==========
    print(f"\n--- Saving ---")
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        temp_path = f.name
    
    tokens_np = np.array(tokens_multi, dtype=np.uint16)
    save_start = time.time()
    np.save(temp_path, tokens_np)
    save_elapsed = time.time() - save_start
    
    file_size = os.path.getsize(temp_path)
    print(f"Saved {len(tokens_np)} tokens to {temp_path}")
    print(f"File size: {file_size / 1024:.2f} KB")
    print(f"Save time: {save_elapsed:.4f}s")
    
    # ========== LOAD ==========
    print(f"\n--- Loading ---")
    load_start = time.time()
    tokens_loaded = np.load(temp_path)
    load_elapsed = time.time() - load_start
    
    print(f"Loaded {len(tokens_loaded)} tokens")
    print(f"Load time: {load_elapsed:.4f}s")
    
    # Check load consistency
    if np.array_equal(tokens_np, tokens_loaded):
        print(f"✓ Save/Load consistency: PASSED")
    else:
        print(f"✗ Save/Load consistency: FAILED")
    
    # ========== DECODE ==========
    print(f"\n--- Decoding ---")
    decode_start = time.time()
    decoded_text = tokenizer.decode(tokens_loaded.tolist())
    decode_elapsed = time.time() - decode_start
    
    print(f"Decoded text length: {len(decoded_text)} characters")
    print(f"Decode time: {decode_elapsed:.4f}s")
    
    # Check decode consistency (roundtrip)
    if decoded_text == original_text:
        print(f"✓ Roundtrip consistency: PASSED (original == decoded)")
    else:
        print(f"✗ Roundtrip consistency: FAILED")
        print(f"  Original length: {len(original_text)}")
        print(f"  Decoded length:  {len(decoded_text)}")
        # Find first difference
        min_len = min(len(original_text), len(decoded_text))
        for i in range(min_len):
            if original_text[i] != decoded_text[i]:
                print(f"  First diff at position {i}:")
                print(f"    Original: {repr(original_text[max(0,i-20):i+20])}")
                print(f"    Decoded:  {repr(decoded_text[max(0,i-20):i+20])}")
                break
    
    # ========== SUMMARY ==========
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  Single-process encode: {single_elapsed:.4f}s")
    print(f"  Multi-process encode:  {multi_elapsed:.4f}s")
    print(f"  Speedup:               {single_elapsed/multi_elapsed:.2f}x")
    print(f"  Save:                  {save_elapsed:.4f}s")
    print(f"  Load:                  {load_elapsed:.4f}s")
    print(f"  Decode:                {decode_elapsed:.4f}s")
    print(f"  Total pipeline:        {multi_elapsed + save_elapsed + load_elapsed + decode_elapsed:.4f}s")
    
    # Cleanup
    os.unlink(temp_path)
    print(f"\nCleaned up temp file: {temp_path}")
    
    return tokens_single == tokens_multi and decoded_text == original_text


def main():
    parser = argparse.ArgumentParser(description='Tokenize OWT data')
    parser.add_argument('--test', action='store_true', 
                        help='Run test mode with 64 sampled documents')
    parser.add_argument('--num-samples', type=int, default=64,
                        help='Number of documents to sample in test mode (default: 64)')
    parser.add_argument('--num-process', type=int, default=4,
                        help='Number of processes for multi-process encoding (default: 4)')
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = TrainedTokenizer.from_files(owt_vocab, owt_merges, special_tokens)
    
    # Start timing
    total_start_time = time.time()
    
    # Test mode: load all documents upfront for sampling
    if args.test:
        print("Loading documents...")
        load_start_time = time.time()
        owt_train_documents = load_documents(owt_train)
        owt_val_documents = load_documents(owt_val)
        load_elapsed = time.time() - load_start_time
        print(f"Loading documents took {load_elapsed:.2f} seconds")
        
        print("\n" + "="*60)
        print("RUNNING IN TEST MODE")
        print("="*60)
        
        # Test with train documents
        print("\n>>> Testing with TRAIN documents <<<")
        train_ok = test_encoding_pipeline(
            tokenizer, 
            owt_train_documents, 
            num_samples=args.num_samples,
            num_process=args.num_process
        )
        
        # Test with val documents
        print("\n>>> Testing with VAL documents <<<")
        val_ok = test_encoding_pipeline(
            tokenizer, 
            owt_val_documents, 
            num_samples=args.num_samples,
            num_process=args.num_process
        )
        
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        print(f"  Train test: {'PASSED ✓' if train_ok else 'FAILED ✗'}")
        print(f"  Val test:   {'PASSED ✓' if val_ok else 'FAILED ✗'}")
        
        total_elapsed = time.time() - total_start_time
        print(f"\nTotal test time: {total_elapsed:.2f} seconds")
        return
    
    # Full tokenization mode
    # Start memory tracking
    tracemalloc.start()
    print("\n[Memory tracking started]")
    
    # Process train: load -> tokenize -> release (minimize memory footprint)
    print("\nLoading train documents...")
    owt_train_documents = load_documents(owt_train)
    print_memory_stats("After loading train documents")
    
    print("\nTokenizing train documents...")
    train_start_time = time.time()
    train_tokens = tokenize_documents_multiprocess(tokenizer, owt_train_documents, num_process=args.num_process)
    del owt_train_documents  # Release memory immediately after tokenization
    train_elapsed = time.time() - train_start_time
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Tokenizing train documents took {train_elapsed:.2f} seconds")
    print_memory_stats("After train tokenization (documents released)")

    # Process val: load -> tokenize -> release
    print("\nLoading val documents...")
    owt_val_documents = load_documents(owt_val)
    print_memory_stats("After loading val documents")
    
    print("\nTokenizing val documents...")
    val_start_time = time.time()
    val_tokens = tokenize_documents_multiprocess(tokenizer, owt_val_documents, num_process=args.num_process)
    del owt_val_documents  # Release memory immediately after tokenization
    val_elapsed = time.time() - val_start_time
    print(f"Val tokens: {len(val_tokens)}")
    print(f"Tokenizing val documents took {val_elapsed:.2f} seconds")
    print_memory_stats("After val tokenization (documents released)")

    # Convert to numpy arrays and save
    train_tokens_np = np.array(train_tokens, dtype=np.uint16)  # uint16 supports vocab up to 65535
    val_tokens_np = np.array(val_tokens, dtype=np.uint16)

    # Save paths
    train_output = os.path.join(data_dir, 'owt_train_tokens.npy')
    val_output = os.path.join(data_dir, 'owt_val_tokens.npy')

    np.save(train_output, train_tokens_np)
    print(f"Saved train tokens to {train_output}")

    np.save(val_output, val_tokens_np)
    print(f"Saved val tokens to {val_output}")
    print_memory_stats("After saving")

    # Get final peak memory
    current_mb, peak_mb = get_peak_memory_mb()
    tracemalloc.stop()

    total_elapsed = time.time() - total_start_time

    print(f"\nSummary:")
    print(f"  Train: {len(train_tokens_np)} tokens, shape {train_tokens_np.shape}, dtype {train_tokens_np.dtype}")
    print(f"  Val:   {len(val_tokens_np)} tokens, shape {val_tokens_np.shape}, dtype {val_tokens_np.dtype}")
    print(f"\nTiming Summary:")
    print(f"  Tokenizing train:  {train_elapsed:.2f} seconds")
    print(f"  Tokenizing val:    {val_elapsed:.2f} seconds")
    print(f"  Total time:        {total_elapsed:.2f} seconds")
    print(f"\nMemory Summary:")
    print(f"  Peak memory usage: {peak_mb:.2f} MB")


# cd /data/home/svenshen/cs336/assignment1-basics && uv run python my_scripts/tokenize_owt_data.py 
if __name__ == "__main__":
    main()