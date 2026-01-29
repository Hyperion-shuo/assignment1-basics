import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
# with open(..., "rb") as f:
#     num_processes = 4
#     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

#     # The following is a serial implementation, but you can parallelize this
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token


def find_chunk_boundaries_for_str(
    text: str,
    desired_num_chunks: int,
    split_special_token: str = "<|endoftext|>",
) -> list[int]:
    """
    针对字符串的切分函数，返回字符索引列表。
    """
    text_len = len(text)
    if text_len == 0:
        return [0]
        
    chunk_size = text_len // desired_num_chunks
    
    # 初始猜测的切分点
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = text_len # 确保最后一点是文本末尾

    mini_chunk_size = 1024  # 每次向后探测 1024 个字符

    # 修正中间的切分点，确保它们落在 special_token 之后
    # 这样可以保证 special_token 不会被切断
    for i in range(1, len(boundaries) - 1):
        initial_pos = boundaries[i]
        current_pos = initial_pos
        
        while True:
            # 向后看一段
            mini_chunk = text[current_pos : current_pos + mini_chunk_size]
            
            if not mini_chunk: # 到头了
                boundaries[i] = text_len
                break
                
            # 查找特殊 token
            found_at = mini_chunk.find(split_special_token)
            
            if found_at != -1:
                # 找到了！新的边界设在 special_token 之后（或者之前，看你需求）
                # 这里假设我们想在 token 之前切，或者 token 之后切都可以
                # 只要保证 token 完整即可。
                # 这里的逻辑是：找到 token 的起始位置，以此作为边界
                boundaries[i] = current_pos + found_at
                break
            
            current_pos += mini_chunk_size
            if current_pos >= text_len:
                boundaries[i] = text_len
                break
    
    # 去重并排序
    return sorted(list(set(boundaries)))