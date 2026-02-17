import time
import resource
import json
from cs336_basics.bpe import BPE


# run with
# scalene run --outfile result.html my_scripts/bpe_train_on_tinystories.py
# uv run my_scripts/bpe_train_on_tinystories.py
if __name__ == "__main__":
    bpe_trainer = BPE(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_process=4,
    )

    print("Starting training...")
    start_time = time.time()
    vocab, merges = bpe_trainer.train()
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")
    # resource.getrusage(resource.RUSAGE_SELF).ru_maxrss return value is in kilobytes on Linux
    max_memory_GB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"Max memory usage: {max_memory_GB:.2f} GB")
    
    # Save vocab as JSON with bytes as list of integers (to handle all byte values correctly)
    # Format: {token_id: [byte1, byte2, ...], ...}
    with open("data/tinystories_bpe_vocab.json", "w", encoding="utf-8") as f:
        vocab_bytes = {k: list(v) for k, v in vocab.items()}
        json.dump(vocab_bytes, f, indent=2)
    
    # Save merges as JSON with bytes as list of integers
    # Format: [[[byte1, byte2], [byte3, byte4]], ...]
    with open("data/tinystories_bpe_merges.json", "w", encoding="utf-8") as f:
        merges_bytes = [[list(merge[0]), list(merge[1])] for merge in merges]
        json.dump(merges_bytes, f, indent=2)
    
    print(f"Saved vocab ({len(vocab)} tokens) to data/tinystories_bpe_vocab.json")
    print(f"Saved merges ({len(merges)} merges) to data/tinystories_bpe_merges.json")