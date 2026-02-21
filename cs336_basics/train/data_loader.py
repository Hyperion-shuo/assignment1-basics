import os
import torch
import numpy as np
import numpy.typing as npt

from collections import deque
from typing import Generator, Tuple
from jaxtyping import Int, Float
from torch import Tensor

def dataloader_sequential(token_ids : npt.NDArray,
               B: Int, 
               T: Int, 
               device: str
    ) -> Generator[Tuple[Tensor, Tensor], None, None]:

    num_samples = len(token_ids)
    needed_tokens = B * T + 1
    current_idx = 0

    while current_idx + needed_tokens <= num_samples:
        batch = torch.from_numpy(token_ids[current_idx:current_idx + needed_tokens]).to(device)
        # Move forward by B*T tokens, to keep the last token for the next batch's start
        current_idx += needed_tokens - 1 
        yield batch[:B*T].reshape(B, T), batch[1:B*T+1].reshape(B, T)

def dataloader(token_ids : npt.NDArray,
               B: Int, 
               T: Int, 
               device: str
    ) -> Generator[Tuple[Tensor, Tensor], None, None]:
    """
    Random sampling dataloader for language modeling.
    
    Each call yields a batch of B sequences, where each sequence's starting
    position is randomly sampled from valid range [0, len(token_ids) - T - 1].
    
    Args:
        token_ids: 1D numpy array of token IDs
        B: batch size
        T: context length (sequence length)
        device: torch device string
    
    Yields:
        (input, target) tuple, both of shape (B, T)
        where target[i] = input[i] shifted by 1
    """
    num_tokens = len(token_ids)
    # Valid starting indices: 0 to num_tokens - T - 1
    # Because we need T tokens for input and 1 more for the last target
    max_start_idx = num_tokens - T - 1
    
    while True:
        # Randomly sample B starting indices
        start_indices = np.random.randint(0, max_start_idx + 1, size=B)
        
        # Build input and target tensors
        inputs = []
        targets = []
        for start in start_indices:
            inputs.append(token_ids[start : start + T])
            targets.append(token_ids[start + 1 : start + T + 1])
        
        x = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
        y = torch.tensor(np.array(targets), dtype=torch.long, device=device)
        
        yield x, y

def get_tokenized_data_files(data_dir: str, data_set_name: str) -> list[str]:
    """
    Get list of tokenized data files for the given dataset.

    Args:
        data_dir: Directory where tokenized files are stored
        data_set_name: Name of the dataset (e.g., 'tinystories', 'owt')
    Returns:
        List of file paths to tokenized data files
    """
    tokenized_data_files = []
    for filename in os.listdir(data_dir):
        if filename.startswith(f"{data_set_name}_train_tokens_") and filename.endswith(".npz"):
            tokenized_data_files.append(os.path.join(data_dir, filename))
    return tokenized_data_files