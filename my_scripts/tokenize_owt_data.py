import pathlib
import os
import time
import numpy as np
from cs336_basics.tokenizer import TrainedTokenizer


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

tokenizer = TrainedTokenizer.from_files(owt_vocab, owt_merges, special_tokens)

# Start timing
total_start_time = time.time()

print("Loading documents...")
load_start_time = time.time()
owt_train_documents = load_documents(owt_train)
owt_val_documents = load_documents(owt_val)
load_elapsed = time.time() - load_start_time
print(f"Loading documents took {load_elapsed:.2f} seconds")


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


# Tokenize train and val documents
print("\nTokenizing train documents...")
train_start_time = time.time()
train_tokens = tokenize_documents(tokenizer, owt_train_documents)
train_elapsed = time.time() - train_start_time
print(f"Train tokens: {len(train_tokens)}")
print(f"Tokenizing train documents took {train_elapsed:.2f} seconds")

print("\nTokenizing val documents...")
val_start_time = time.time()
val_tokens = tokenize_documents(tokenizer, owt_val_documents)
val_elapsed = time.time() - val_start_time
print(f"Val tokens: {len(val_tokens)}")
print(f"Tokenizing val documents took {val_elapsed:.2f} seconds")

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

total_elapsed = time.time() - total_start_time

print(f"\nSummary:")
print(f"  Train: {len(train_tokens_np)} tokens, shape {train_tokens_np.shape}, dtype {train_tokens_np.dtype}")
print(f"  Val:   {len(val_tokens_np)} tokens, shape {val_tokens_np.shape}, dtype {val_tokens_np.dtype}")
print(f"\nTiming Summary:")
print(f"  Loading documents: {load_elapsed:.2f} seconds")
print(f"  Tokenizing train:  {train_elapsed:.2f} seconds")
print(f"  Tokenizing val:    {val_elapsed:.2f} seconds")
print(f"  Total time:        {total_elapsed:.2f} seconds")