import torch
from jaxtyping import Float, Int
from torch.distributions.categorical import Categorical
from cs336_basics.tokenizer import TrainedTokenizer
from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.transformer.core import softmax

# current impelementation only support none-batch mode
def generate(transformer: TransformerLM, 
             tokenizer: TrainedTokenizer, 
             prompt: str | list[str], 
             max_length: Int=100, 
             temperature: Float=1.0, 
             top_p: Float=0.9, 
             device: str='cuda'):
    
    token_ids = tokenizer.encode(prompt)
    token_ids = torch.tensor(token_ids, device=device)
    generate_length = 0
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    transformer.eval()
    transformer.to(device)

    with torch.no_grad():
        while generate_length < max_length:
            token_positions = torch.arange(len(token_ids), device=device).unsqueeze(0)
            logits = transformer(token_ids.unsqueeze(0), token_positions)
            last_logits = logits[0, -1, :]
            last_logits /= temperature
            probs = softmax(last_logits)

            sorted_probs, sorted_index = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # not cumulative_probs > topp
            # to reserve the last token cumulate to top_p
            mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            dist = Categorical(sorted_probs)
            sampled_idx = dist.sample()
            new_token_id = sorted_index[sampled_idx]
            if new_token_id == eos_token_id:
                break

            token_ids = torch.cat([token_ids, new_token_id.unsqueeze(0)], dim=-1)
            generate_length += 1
    
    results = tokenizer.decode(token_ids.tolist())

    return results

    


