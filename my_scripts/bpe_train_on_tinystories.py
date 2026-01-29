import time
import resource
from cs336_basics.bpe import BPE


# run with
# scalene run --outfile result.html my_scripts/bpe_train_on_tinystories.py
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

    # print("Vocab:")
    # for k, v in vocab.items():
    #     print(f"{k}: {v}")
    # print("\nMerges:")
    # for merge in merges:
    #     print(merge)
    
    # save vocab and merges to files
    with open("data/tinystories_bpe_vocab.json", "w", encoding="utf-8") as f:
        import json
        json.dump({k: v.decode("utf-8", errors="ignore") for k, v in vocab.items()}, f, ensure_ascii=False, indent=4)
    with open("data/tinystories_bpe_merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{merge[0].decode('utf-8', errors='ignore')} {merge[1].decode('utf-8', errors='ignore')}\n")