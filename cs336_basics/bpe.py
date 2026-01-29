import regex as re
import os
import heapq

from cs336_basics.pretokenization_example import find_chunk_boundaries, find_chunk_boundaries_for_str
from multiprocessing import Pool
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_INVERT_TABLE = bytes.maketrans(bytes(range(256)), bytes(255 - i for i in range(256)))

class Node:
    def __init__(self, token_id):
        self.token_id = token_id
        self.prev = None
        self.next = None
        
    def __repr__(self):
        return f"Node({self.token_id})"

def _process_chunk_re(input_path, start, end, pattern_str, special_tokens):
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        
    text = chunk_bytes.decode("utf-8", errors="ignore")
    # <|endoftext|> convert to \<\|endoftext\|\>, otherwise | will be misunderstand as 'or'
    escaped_special_tokens = [re.escape(t) for t in special_tokens]
    special_pattern = "|".join(escaped_special_tokens)
    splits = re.split(special_pattern, text)
    
    frequency_table = Counter()
    bpe_pattern = re.compile(pattern_str)
    for split in splits:
        # skip start or end empty split str
        if not split:
            continue
        tokens = bpe_pattern.findall(split)
        for i in range(len(tokens)):
            cur_token = tuple(tokens[i].encode("utf-8"))
            frequency_table[cur_token] += 1 
        
    return frequency_table

# Naive count without cache pair position 
# def get_status(frequency_table):
#     count = defaultdict(int)
#     for token, freq in frequency_table.items():
#         if len(token) < 2:
#             continue
#         for pair in zip(token[:-1], token[1:]):
#             count[pair] += freq  
#     return count

# Naive merge without cache pair position 
# def merge(frequency_table, pair, idx):
#     new_frequency_table = {}
#     for token_ids, freq in frequency_table.items():
#         i = 0
#         new_token_ids = []
#         while i < len(token_ids):
#             if token_ids[i] == pair[0] and i < len(token_ids) - 1 and token_ids[i+1] == pair[1]:
#                 new_token_ids.append(idx)
#                 i += 2
#             else:
#                 new_token_ids.append(token_ids[i])
#                 i += 1
#         new_frequency_table[tuple(new_token_ids)] = freq
#     return new_frequency_table

def get_status(pair_indexes):
    count = defaultdict(int)
    for pair, pair_ids in pair_indexes.items():
        count[pair] = len(pair_ids)
    return count

class BPE:
    def __init__(self, 
                input_path: str | os.PathLike, 
                vocab_size: int, 
                special_tokens: list[str],
                **kwargs):
        self.vocab_size = vocab_size
        self.input_path = input_path
        self.special_tokens = special_tokens
        self.merges = []
        self.pattern_str = PAT
        # self.pattern = re.compile(PAT)
        self.num_process = kwargs.get("num_process", 16)
    
    def _build_vocab(self):
        # must add [], bytes(x) return zero bytes with length x
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def _build_double_linked_list(self, frequency_table):
        self.words = []
        self.pair_indexes = defaultdict(list)
        
        for word_idx, (token_ids, freq) in enumerate(frequency_table.items()):
            if len(token_ids) == 0:
                continue
            
            head = Node(token_ids[0])
            curr = head
            if len(token_ids) > 1:  
                for i in range(1, len(token_ids)):
                    new_node = Node(token_ids[i])
                    curr.next = new_node
                    new_node.prev = curr
                    
                    pair = (curr.token_id, new_node.token_id)
                    self.pair_indexes[pair].append((word_idx, curr))
                    curr = new_node
                   
            self.words.append({'head': head,
                               'freq': freq})

    def _heap_key(self, pair, count):
        b0 = self.vocab[pair[0]].translate(_INVERT_TABLE)
        b1 = self.vocab[pair[1]].translate(_INVERT_TABLE)
        return (-count, b0, b1, pair)
            
    def _merge_cached(self, pair, new_token_id):
        bigram0, bigram1 = pair
        locations = self.pair_indexes[pair]
        updated_pairs = set()
        
        for word_idx, node in locations:
            # Skip if this location has already been invalidated by a previous merge
            if node.token_id != bigram0 or node.next is None or node.next.token_id != bigram1:
                continue
            
            word_freq = self.words[word_idx]['freq']
            node_to_remove = node.next
            
            # Decrease count for (prev, node) - this pair will be replaced
            if node.prev:
                prev_pair = (node.prev.token_id, node.token_id)
                self.count[prev_pair] -= word_freq
                updated_pairs.add(prev_pair)
            
            # Decrease count for (node_to_remove, next) - this pair will be replaced
            if node_to_remove.next:
                next_pair = (node_to_remove.token_id, node_to_remove.next.token_id)
                self.count[next_pair] -= word_freq
                updated_pairs.add(next_pair)
            
            # Perform the merge: node becomes the merged token
            node.token_id = new_token_id
            node.next = node_to_remove.next
            if node_to_remove.next:
                node_to_remove.next.prev = node
            
            # IMPORTANT: Invalidate the removed node to prevent stale references
            # in pair_indexes from causing incorrect counts
            node_to_remove.token_id = -1  # Invalid token_id
            node_to_remove.next = None
            node_to_remove.prev = None
            
            # Increase count for new (prev, merged_node) pair
            if node.prev:
                new_prev_pair = (node.prev.token_id, node.token_id)
                self.pair_indexes[new_prev_pair].append((word_idx, node.prev))
                self.count[new_prev_pair] += word_freq
                updated_pairs.add(new_prev_pair)
                
            # Increase count for new (merged_node, next) pair
            if node.next:
                new_next_pair = (node.token_id, node.next.token_id)
                self.pair_indexes[new_next_pair].append((word_idx, node))
                self.count[new_next_pair] += word_freq
                updated_pairs.add(new_next_pair)
        
        del self.pair_indexes[pair]
        del self.count[pair]
        updated_pairs.discard(pair)
        return updated_pairs

    
    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        assert self.vocab_size >= 256 + len(self.special_tokens)
        self.num_merges = self.vocab_size - (256 + len(self.special_tokens))
        self._build_vocab()
        
        # with open(self.input_path, 'r', encoding='utf-8') as f:
        with open(self.input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, self.num_process, b"<|endoftext|>")

        # you can parallelize this by sending each start/end pair to a set of processes.
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((self.input_path, start, end, self.pattern_str, self.special_tokens))
        
        # removing special tokens by re.split
        # TODO: use finditer, now _process_chunk_re use find all
        frequency_table = Counter()
        with Pool(processes=self.num_process) as pool:
            results = pool.starmap(_process_chunk_re, tasks)
            for freq_table in results:
                for token, count in freq_table.items():
                    frequency_table[token] = frequency_table.get(token, 0) + count
        self._build_double_linked_list(frequency_table)  
                 
        new_token_id = 256
        self.count = defaultdict(int)
        for pair, locations in self.pair_indexes.items():
            for word_idx, node in locations:
                self.count[pair] += self.words[word_idx]['freq']
        for i in range(self.num_merges): 
            # break ties in pair frequency by preferring the lexicographically greater pair (by bytes)
            # bytes can be directly compared in Python using lexicographic order
            max_pair, max_count = max(self.count.items(), key=lambda x: (x[1], self.vocab[x[0][0]], self.vocab[x[0][1]]))
            # merged_frequency_table = merge(frequency_table, max_pair, idx)
            # frequency_table = merged_frequency_table
            self._merge_cached(max_pair, new_token_id)
            self.merges.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))
            self.vocab[new_token_id] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            new_token_id += 1

        # add special to vocab at the end
        for special_token in self.special_tokens:
            self.vocab[new_token_id] = bytes(special_token.encode("utf-8"))
            new_token_id += 1
            
        return self.vocab, self.merges  

# If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocab-
# ulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].

# run with uv run python cs336_basics/bpe_trainer.py under assignment1-basics
if __name__ == "__main__":
    bpe_trainer = BPE(
        input_path="tests/fixtures/bpe_example_test.txt",
        vocab_size=263,
        special_tokens=["<|endoftext|>"],
        num_process=4,
    )
    vocab, merges = bpe_trainer.train()
    # print("Vocab:")
    # for k, v in vocab.items():
    #     print(f"{k}: {v}")
    print("\nMerges:")
    for merge in merges:
        print(merge)
        
# results:
# Merges:
# (b's', b't')
# (b'e', b'st')
# (b'o', b'w')
# (b'l', b'ow')
# (b'w', b'est')
# (b'n', b'e')