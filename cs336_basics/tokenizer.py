import token
from turtle import up
import regex as re
import os
import heapq

from tqdm import tqdm
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from collections import Counter, defaultdict
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Node:
    def __init__(self, token_id):
        self.token_id = token_id
        self.version = 0
        self.prev = None
        self.next = None
        
    def __lt__(self, other):
        return False  # We don't care about the order of nodes themselves in the heap
        
    def __repr__(self):
        return f"Node({self.token_id})"

def _process_chunk_re(text, pattern_str, special_tokens):
    bpe_pattern = re.compile(pattern_str)
    chunk_list = []
    if not special_tokens:
        tokens = bpe_pattern.findall(text)
        chunk_list.extend((False, token.encode("utf-8")) for token in tokens)
        return chunk_list

    # <|endoftext|> convert to \<\|endoftext\|\>, otherwise | will be misunderstand as 'or'
    escaped_special_tokens = [re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)]
    special_pattern = "|".join(escaped_special_tokens)
    splits = re.split(f"({special_pattern})", text)

    for split in splits:
        # skip start or end empty split str
        if not split:
            continue
        if split in special_tokens:
            chunk_list.append((True, split.encode("utf-8")))
            continue
        tokens = bpe_pattern.findall(split)
        chunk_list.extend((False, token.encode("utf-8")) for token in tokens)

    return chunk_list

def _worker_encode(text_chunk: str, vocab: dict[int, bytes], inv_vocab: dict[bytes, int], 
                   merges_id: dict[tuple[int, int], int], pattern_str: str, 
                   special_tokens: list[str] | None) -> list[int]:
    """
    Worker function for multi-process encoding.
    This function runs in a separate process and encodes a single text chunk.
    
    Args:
        text_chunk: The text chunk to encode
        vocab: Token ID to bytes mapping
        inv_vocab: Bytes to token ID mapping
        merges_id: Merge pair to merge order mapping
        pattern_str: Regex pattern string for tokenization
        special_tokens: List of special tokens
    
    Returns:
        List of token IDs
    """
    # Process the text chunk to get byte chunks
    chunks_bytes_list = _process_chunk_re(text_chunk, pattern_str, special_tokens if special_tokens else [])
    
    encode_ids = []
    for is_special, chunks_bytes in chunks_bytes_list:
        if is_special:
            token_id = inv_vocab.get(chunks_bytes)
            if token_id is None:
                raise ValueError(f"Special token not found in vocab: {chunks_bytes!r}")
            encode_ids.append(token_id)
            continue
        
        # Build double linked list for BPE merging
        if len(chunks_bytes) == 0:
            continue
            
        # Build linked list
        head = Node(inv_vocab.get(bytes([chunks_bytes[0]])))
        curr = head
        chunk_heap = []
        
        if len(chunks_bytes) > 1:
            for i in range(1, len(chunks_bytes)):
                new_node = Node(inv_vocab.get(bytes([chunks_bytes[i]])))
                curr.next = new_node
                new_node.prev = curr
                pair = (curr.token_id, new_node.token_id)
                heapq.heappush(chunk_heap, (merges_id.get(pair, float('inf')), pair, curr))
                curr = new_node
        
        # Perform BPE merges
        while chunk_heap:
            merge_id, pair, node = heapq.heappop(chunk_heap)
            if merge_id == float('inf'):
                break
            
            bigram0, bigram1 = pair
            # Skip if this location has already been invalidated
            if node.token_id != bigram0 or node.next is None or node.next.token_id != bigram1:
                continue
            
            node_to_remove = node.next
            # Perform the merge
            merged_bytes = vocab[bigram0] + vocab[bigram1]
            new_token_id = inv_vocab.get(merged_bytes)
            if new_token_id is None:
                raise ValueError(f"Merged token not in vocab: {merged_bytes!r}")
            node.token_id = new_token_id
            node.next = node_to_remove.next
            if node_to_remove.next:
                node_to_remove.next.prev = node
            
            # Invalidate the removed node
            node_to_remove.token_id = -1
            node_to_remove.next = None
            node_to_remove.prev = None
            
            # Add new pairs to heap
            if node.prev:
                new_prev_pair = (node.prev.token_id, node.token_id)
                heapq.heappush(chunk_heap, (merges_id.get(new_prev_pair, float('inf')), new_prev_pair, node.prev))
            if node.next:
                new_next_pair = (node.token_id, node.next.token_id)
                heapq.heappush(chunk_heap, (merges_id.get(new_next_pair, float('inf')), new_next_pair, node))
        
        # Extract tokens from linked list
        curr = head
        while curr:
            encode_ids.append(curr.token_id)
            curr = curr.next
    
    return encode_ids


def _worker_encode_tuple(args):
    """
    Wrapper for _worker_encode that accepts a single tuple argument.
    This is needed for use with Pool.imap which only accepts single-argument functions.
    """
    return _worker_encode(*args)


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
        import json
        
        # Load vocab from JSON file
        # Format: {token_id: [byte1, byte2, ...], ...}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
        # Convert vocab: {str(token_id): list[int]} -> {int: bytes}
        vocab = {int(k): bytes(v) for k, v in vocab_data.items()}
        
        # Load merges from JSON file
        # Format: [[[byte1, byte2], [byte3, byte4]], ...]
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_data = json.load(f)
        
        # Convert merges: list of [list[int], list[int]] -> list of (bytes, bytes)
        merges = [(bytes(merge[0]), bytes(merge[1])) for merge in merges_data]
        
        return cls(vocab, merges, special_tokens)
    
    def _build_double_linked_list(self, chunk_bytes: bytes):
        if len(chunk_bytes) == 0:
            return {'head': None, 'heap': []}

        head = Node(self._byte_to_token_id(chunk_bytes[0]))
        curr = head
        chunk_heap = []
        if len(chunk_bytes) > 1:  
            for i in range(1, len(chunk_bytes)):
                new_node = Node(self._byte_to_token_id(chunk_bytes[i]))
                curr.next = new_node
                new_node.prev = curr
                pair = (curr.token_id, new_node.token_id)
                # TODO: tie break
                heapq.heappush(chunk_heap, self._heap_key(self.merges_id.get(pair, float('inf')), pair, curr))
                curr = new_node
                
        return {'head': head, 'heap': chunk_heap}

    def _byte_to_token_id(self, byte_val: int) -> int:
        token_id = self.inv_vocab.get(bytes([byte_val]))
        if token_id is None:
            raise ValueError(f"Unknown byte in vocab: {byte_val}")
        return token_id
    
    def _extract_tokens_from_linked_list(self, head: Node) -> list[int]:
        tokens = []
        curr = head
        while curr:
            tokens.append(curr.token_id)
            curr = curr.next
        return tokens
            
    def _heap_key(self, merge_id, pair, node):
        return (merge_id, pair, node)
    
    
    def _merge(self, pair, node):
        bigram0, bigram1 = pair
        updated_pairs = []
        
        # Skip if this location has already been invalidated by a previous merge
        # TODO: check if this is sufficient, other wise using versioning
        if node.token_id != bigram0 or node.next is None or node.next.token_id != bigram1:
            return updated_pairs
        
        node_to_remove = node.next
        # Perform the merge: node becomes the merged token
        merged_bytes = self.vocab[bigram0] + self.vocab[bigram1]
        new_token_id = self.inv_vocab.get(merged_bytes)
        if new_token_id is None:
            raise ValueError(f"Merged token not in vocab: {merged_bytes!r}")
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
            updated_pairs.append((self.merges_id.get(new_next_pair, float('inf')), new_next_pair, node))
        
        return updated_pairs
    
    def encode(self, text: str) -> list[int]:
        # if "<|endoftext|>" in text and self.special_tokens and "<|endoftext|>" in self.special_tokens:
        #     print('Multi process encoding...')
        #     boundaries = find_chunk_boundaries_for_str(text, self.num_process, "<|endoftext|>")
            
        #     tasks = []
        #     for i in range(len(boundaries) - 1):
        #         start = boundaries[i]
        #         end = boundaries[i+1]
        #         chunk = text[start:end]
        #         if chunk:
        #             # 这里的参数要和 worker_encode 对应
        #             tasks.append((chunk, r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", self.special_tokens))

        #     # 3. 多进程执行
        #     chunks_bytes_list = []
        #     with Pool(processes=self.num_process) as pool:
        #         # 使用 starmap 因为 worker 接收多个参数
        #         results = pool.starmap(_process_chunk_re, tasks)
                
        #         # 4. 合并结果
        #         for res in results:
        #             chunks_bytes_list.extend(res)
        # else:
        #     print('Single process encoding...')
        
        chunks_bytes_list = _process_chunk_re(text, self.pattern_str, self.special_tokens if self.special_tokens else [])

        encode_ids = []
        for is_special, chunks_bytes in chunks_bytes_list:
            if is_special:
                token_id = self.inv_vocab.get(chunks_bytes)
                if token_id is None:
                    raise ValueError(f"Special token not found in vocab: {chunks_bytes!r}")
                encode_ids.append(token_id)
                continue
            chunk = self._build_double_linked_list(chunks_bytes)
            head, chunk_heap = chunk['head'], chunk['heap']
            while chunk_heap:
                merge_id, pair, node = heapq.heappop(chunk_heap)
                if merge_id == float('inf'):
                    break
                updated_pairs = self._merge(pair, node)
                for updated_pair in updated_pairs:
                    heapq.heappush(chunk_heap, updated_pair)
            encode_ids.extend(self._extract_tokens_from_linked_list(head))
        return encode_ids

    def encode_file_streaming(self, filepath: str, num_chunks: int = 16) -> Iterator[list[int]]:
        """
        Stream-encode a file: yields token lists chunk by chunk.
        Uses find_chunk_boundaries for memory-efficient boundary detection.
        
        Args:
            filepath: Path to text file
            num_chunks: Number of chunks to split file into
            
        Yields:
            List of token IDs for each chunk
        """
        delimiter = b"<|endoftext|>"
        
        with open(filepath, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, delimiter)
            
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                
                if chunk_text:
                    yield self.encode(chunk_text)

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