import token
import regex as re
import os
import heapq

from tqdm import tqdm
from cs336_basics.pretokenization_example import find_chunk_boundaries, find_chunk_boundaries_for_str
from multiprocessing import Pool
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def navie_get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def navie_merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def worker_encode(text_chunk, pattern_str, merges_id, special_tokens):
    """
    子进程的工作函数：
    1. 接收文本片段
    2. 使用正则进行细粒度分割
    3. 执行 BPE 编码
    """
    
    # split special tokens
    # <|endoftext|> convert to \<\|endoftext\|\>, otherwise | will be misunderstand as 'or'
    escaped_special_tokens = [re.escape(t) for t in special_tokens]
    special_pattern = "|".join(escaped_special_tokens)
    splits = re.split(special_pattern, text_chunk)
    
    # split pre-tokenization
    pat = re.compile(pattern_str)
    raw_tokens = []
    for split in splits:
        if not split:
            continue
        raw_tokens.extend(re.findall(pat, split))
    
    all_ids = []
    
    # 2. 对每个小片段进行 BPE 编码
    for token_text in raw_tokens:
        # 这里需要把 str 转为 bytes 进行 BPE 处理，因为 BPE 是基于 bytes 的
        token_bytes = token_text.encode("utf-8")
        
        new_token_ids = []
        while len(token_bytes) > 2:
            stat = navie_get_stats(list(token_bytes))
        all_ids.extend(new_token_ids)
        
    return all_ids
  
class TrainedTokenizer:
    def __init__(self, vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str]| None = None):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_id = [(self.inv_vocab[merge[0]], self.inv_vocab[merge[1]]) for merge in merges]
        self.special_tokens = special_tokens
        self.pattern_str = PAT
        self.pattern = re.compile(PAT)
        self.num_process = 16
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass
    
    # TODO use heapq and double linked list like train
    def encode(self, text: str) -> list[int]:
        # TODO: special tokens handling
        if "<|endoftext|>" in text and self.special_tokens and "<|endoftext|>" in self.special_tokens:
            boundaries = find_chunk_boundaries_for_str(text, self.num_process, "<|endoftext|>")
            
            tasks = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i+1]
                chunk = text[start:end]
                if chunk:
                    # 这里的参数要和 worker_encode 对应
                    tasks.append((chunk, r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", self.vocab, self.merges))

            # 3. 多进程执行
            final_ids = []
            with Pool(processes=self.num_process) as pool:
                # 使用 starmap 因为 worker 接收多个参数
                results = pool.starmap(worker_encode, tasks)
                
                # 4. 合并结果
                for res in results:
                    final_ids.extend(res)
            
            return final_ids
        else:
            return []
            
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in tqdm(iterable):
            yield from self.encode(line)        
    
    def decode(self, ids: list[int]) -> str:
        
    
if __name__ == "__main__":
    pass