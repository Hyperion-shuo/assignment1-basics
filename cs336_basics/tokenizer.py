import token
from turtle import up
import regex as re
import os
import heapq

from tqdm import tqdm
from cs336_basics.pretokenization_example import find_chunk_boundaries, find_chunk_boundaries_for_str
from multiprocessing import Pool
from collections import Counter, defaultdict
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class _RevBytes:
    __slots__ = ("value",)

    def __init__(self, value: bytes):
        self.value = value

    def __lt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

class Node:
    def __init__(self, token_id):
        self.token_id = token_id
        self.version = 0
        self.prev = None
        self.next = None
        
    def __repr__(self):
        return f"Node({self.token_id})"

def _process_chunk_re(text, pattern_str, special_tokens):
    # <|endoftext|> convert to \<\|endoftext\|\>, otherwise | will be misunderstand as 'or'
    escaped_special_tokens = [re.escape(t) for t in special_tokens]
    special_pattern = "|".join(escaped_special_tokens)
    splits = re.split(special_pattern, text)
    
    bpe_pattern = re.compile(pattern_str)
    chunk_list = []
    for split in splits:
        # skip start or end empty split str
        if not split:
            continue
        tokens = bpe_pattern.findall(split)
        # TODO: check if this is list in list
        chunk_list.extend(tokens.encode('utf-8') for token in tokens)
        
    return chunk_list

def get_status(pair_indexes):
    count = defaultdict(int)
    for pair, pair_ids in pair_indexes.items():
        count[pair] = len(pair_ids)
    return count
  
class TrainedTokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_id = {(self.inv_vocab[merge[0]], self.inv_vocab[merge[1]]): i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens
        self.pattern_str = PAT
        self.pattern = re.compile(PAT)
        self.num_process = 16
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass
    
    def _build_double_linked_list(self, chunk_bytes: bytes):
        if len(chunk_bytes) == 0:
            return {'head': None, 'heap': []}
        
        head = Node(chunk_bytes[0])
        curr = head
        chunk_heap = []
        if len(chunk_bytes) > 1:  
            for i in range(1, len(chunk_bytes)):
                new_node = Node(chunk_bytes[i])
                curr.next = new_node
                new_node.prev = curr
                pair = (curr.token_id, new_node.token_id)
                merge_id = self.merges_id[pair] if pair in self.merges_id else float('inf')
                heapq.heappush(chunk_heap, self._heap_key(merge_id, pair, curr))
                curr = new_node
                
        return {'head': head,' heap': chunk_heap}
    
    def _extract_tokens_from_linked_list(self, head: Node) -> list[int]:
        tokens = []
        curr = head
        while curr:
            tokens.append(curr.token_id)
            curr = curr.next
        return tokens
            
    def _heap_key(self, merge_id, pair, node):
        return (merge_id, pair, node)
    
    
    def _merge(self, pair, merge_id, node):
        new_token_id = merge_id + 256  # Offset to avoid collision with existing token IDs
        bigram0, bigram1 = pair
        updated_pairs = []
        
        # Skip if this location has already been invalidated by a previous merge
        if node.token_id != bigram0 or node.next is None or node.next.token_id != bigram1:
            return updated_pairs
        
        node_to_remove = node.next
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
        
        if node.prev:
            new_prev_pair = (node.prev.token_id, node.token_id)
            updated_pairs.append((self.merges_id.get(new_prev_pair, float('inf')), new_prev_pair, node.prev))
            
        if node.next:
            new_next_pair = (node.token_id, node.next.token_id)
            updated_pairs.append((self.merges_id.get(new_next_pair, float('inf')), new_next_pair, node.next))
        
        return updated_pairs
    
    # TODO use heapq and double linked list like train
    def encode(self, text: str) -> list[int]:
        if "<|endoftext|>" in text and self.special_tokens and "<|endoftext|>" in self.special_tokens:
            print('Multi process encoding...')
            boundaries = find_chunk_boundaries_for_str(text, self.num_process, "<|endoftext|>")
            
            tasks = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i+1]
                chunk = text[start:end]
                if chunk:
                    # 这里的参数要和 worker_encode 对应
                    tasks.append((chunk, r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", self.special_tokens))

            # 3. 多进程执行
            chunks_bytes_list = []
            with Pool(processes=self.num_process) as pool:
                # 使用 starmap 因为 worker 接收多个参数
                results = pool.starmap(_process_chunk_re, tasks)
                
                # 4. 合并结果
                for res in results:
                    chunks_bytes_list.extend(res)
        else:
            print('Single process encoding...')
            chunks_bytes_list = _process_chunk_re(text, self.pattern_str, self.special_tokens if self.special_tokens else [])
        
        encode_ids = []
        for chunks_bytes in chunks_bytes_list:
            chunk = self._build_double_linked_list(chunks_bytes)
            head, chunk_heap = chunk['head'], chunk['heap']
            while chunk_heap:
                merge_id, pair, node = heapq.heappop(chunk_heap)
                updated_pairs = self._merge(pair, merge_id, node)
                for updated_pair in updated_pairs:
                    heapq.heappush(chunk_heap, updated_pair)
            encode_ids.extend(self._extract_tokens_from_linked_list(head))
        return encode_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in tqdm(iterable):
            yield from self.encode(line)        
    
    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        
        # In the case that the input token IDs do not produce a valid Unicode string, 
        # you should replace the malformed bytes with the oﬀicial Unicode replacement character U+FFFD.
        for cur_id in ids:
            if cur_id not in self.vocab:
                byte_parts.append(b"\xef\xbf\xbd")  # U+FFFD in UTF-8
            else:
                byte_parts.append(self.vocab[cur_id])
        
        return b"".join(byte_parts).decode("utf-8", errors="replace")
        
    
if __name__ == "__main__":
    pass