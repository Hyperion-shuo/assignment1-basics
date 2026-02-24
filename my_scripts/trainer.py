import os
import time
import numpy.typing as npt
import torch
import wandb
import argparse
import numpy as np

from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import linalg as LA
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.tokenizer import TrainedTokenizer
from cs336_basics.train.data_loader import dataloader, dataloader_sequential, get_tokenized_data_files
from cs336_basics.train.optimizer import AdamW, lr_cos_schedule, gradient_clipping
from cs336_basics.train.loss import cross_entropy_loss, perplexity
from cs336_basics.train.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.generate import generate


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Language Model Trainer")

    # tokenizer and dataset
    parser.add_argument("--data_set_name", type=str, default="tinystories", choices=["tinystories", "owt"])
    parser.add_argument("--vocab_filepath", type=str, default=None, help="Path to vocab json. Auto-inferred from data_set_name if not set.")
    parser.add_argument("--merges_filepath", type=str, default=None, help="Path to merges json. Auto-inferred from data_set_name if not set.")
    parser.add_argument("--special_tokens", type=str, nargs="+", default=["<|endoftext|>"])
    parser.add_argument("--vocab_size", type=int, default=10000)

    # model
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--use_rope", action=argparse.BooleanOptionalAction, default=True, help="Use RoPE (default: True). Use --no-use_rope to disable.")
    parser.add_argument("--pre_norm", action=argparse.BooleanOptionalAction, default=True, help="Use pre-norm (default: True). Use --no-pre_norm for post-norm.")
    parser.add_argument("--with_rms_norm", action=argparse.BooleanOptionalAction, default=True, help="Use RMSNorm (default: True). Use --no-with_rms_norm to disable.")

    # train loop
    parser.add_argument("--total_tokens", type=int, default=327680000, help="Total number of tokens to train on. Default 327680000 (~5000 iters with default batch_size and context_length).")
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--save_num_iter", type=int, default=500)
    parser.add_argument("--log_num_iter", type=int, default=100)
    parser.add_argument("--generate_num_iter", type=int, default=500)
    parser.add_argument("--generate_prompt", type=str, default="Once upon a time")
    parser.add_argument("--generate_max_length", type=int, default=100)
    parser.add_argument("--eval_num_iter", type=int, default=500)
    parser.add_argument("--eval_num_batches", type=int, default=20)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable wandb logging")
    parser.add_argument("--wandb_project_name", type=str, default=None, help="Auto-inferred if not set.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Auto-inferred if not set.")

    # hyper params
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--alpha_min", type=float, default=1e-5)
    parser.add_argument("--alpha_max", type=float, default=3e-5)
    parser.add_argument("--t_w", type=int, default=500)
    parser.add_argument("--t_c", type=int, default=5000)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999], help="AdamW betas, e.g. 0.9 0.95")
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Auto-inferred if not set.")
    parser.add_argument("--checkpoint_load_path", type=str, default=None, help="Auto-inferred if not set. Set to '' to disable loading.")

    args = parser.parse_args()

    # handle --no_wandb flag
    if args.no_wandb:
        args.use_wandb = False

    # auto-infer paths based on data_set_name
    if args.vocab_filepath is None:
        args.vocab_filepath = os.path.join(base_dir, 'data', f'{args.data_set_name}_bpe_vocab.json')
    if args.merges_filepath is None:
        args.merges_filepath = os.path.join(base_dir, 'data', f'{args.data_set_name}_bpe_merges.json')
    if args.wandb_project_name is None:
        args.wandb_project_name = f"cs336_basics_{args.data_set_name}"
    if args.wandb_run_name is None:
        args.wandb_run_name = f"transformer_d{args.d_model}_h{args.num_heads}_l{args.num_layers}_ff{args.d_ff}_b128"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(base_dir, 'models', args.data_set_name)
    if args.checkpoint_load_path is None:
        args.checkpoint_load_path = os.path.join(base_dir, 'models', args.data_set_name)

    return args


# helper: compute gradient L2 norm (before clipping)
def get_grad_norm(params):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([LA.vector_norm(g, ord=2) for g in grads])
    return LA.vector_norm(norms).item()


# helper: compute total parameter L2 norm
def get_param_norm(params):
    norms = [LA.vector_norm(p.data, ord=2) for p in params if p.data is not None]
    if not norms:
        return 0.0
    return LA.vector_norm(torch.stack(norms)).item()


@torch.no_grad()
def evaluate(model, val_token_ids, args):
    """Evaluate model on validation set, return avg loss and perplexity."""
    model.eval()
    val_loader = dataloader(
        token_ids=val_token_ids,
        B=args.batch_size,
        T=args.context_length,
        device='cuda',
    )
    total_loss = 0.0
    total_ppl = 0.0
    for i, (x, y) in enumerate(val_loader):
        if i >= args.eval_num_batches:
            break
        x, y = x.long(), y.long()
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        ppl = perplexity(logits, y)
        total_loss += loss.item()
        total_ppl += ppl.item()
    avg_loss = total_loss / args.eval_num_batches
    avg_ppl = total_ppl / args.eval_num_batches
    model.train()
    return avg_loss, avg_ppl


def main():
    args = parse_args()

    tokenizer = TrainedTokenizer.from_files(args.vocab_filepath, args.merges_filepath, args.special_tokens)

    transformer = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rope=args.use_rope,
        pre_norm=args.pre_norm,
        with_rms_norm=args.with_rms_norm,
    ).to('cuda')

    optimizer = AdamW(transformer.parameters())

    current_step = 0

    # load val data
    val_token_path = os.path.join(base_dir, 'data', f'{args.data_set_name}_val_tokens.npy')
    val_token_ids = np.load(val_token_path)
    print(f"Loaded val tokens: {len(val_token_ids)}")

    if args.use_wandb:
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args))

    # compute max iters from total_tokens
    tokens_per_iter = args.batch_size * args.context_length
    max_iters = args.total_tokens // tokens_per_iter
    print(f"Training for {max_iters} iters ({args.total_tokens} total tokens, {tokens_per_iter} tokens/iter)")

    done = False
    for _ in range(args.num_epoch):
        if done:
            break
        dataloader_fn = dataloader_sequential
        tokenized_data_files = get_tokenized_data_files(os.path.join(base_dir, 'data'), args.data_set_name)
        for tokenized_data_file in tokenized_data_files:
            if done:
                break
            with open(tokenized_data_file, 'rb') as f:
                token_ids = np.load(f)['tokens']
            train_dataloader = dataloader_fn(
                token_ids=token_ids,
                B=args.batch_size,
                T=args.context_length,
                device='cuda',
            )

            for x, y in train_dataloader:
                if current_step >= max_iters:
                    done = True
                    break
                step_start_time = time.time()
                transformer.train()
                x, y = x.cuda().long(), y.cuda().long()

                # on log steps, also collect per-layer activation norms
                is_log_step = (current_step % args.log_num_iter == 0)
                if is_log_step:
                    logits, layer_norms = transformer(x, return_layer_norms=True)
                else:
                    logits = transformer(x)
                    layer_norms = None
                loss = cross_entropy_loss(logits, y)

                cur_lr = lr_cos_schedule(current_step, args.alpha_min, args.alpha_max, args.t_w, args.t_c)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr
                optimizer.zero_grad()
                loss.backward()
                grad_norm = get_grad_norm(transformer.parameters())
                gradient_clipping(transformer.parameters(), args.max_grad_norm)
                optimizer.step()

                # compute step timing and throughput
                step_time = time.time() - step_start_time
                tokens_per_step = args.batch_size * args.context_length
                throughput = tokens_per_step / step_time  # tokens/sec

                # compute grad clip ratio and param norm
                grad_clip_ratio = grad_norm / args.max_grad_norm
                param_norm = get_param_norm(transformer.parameters())

                if is_log_step:
                    ppl = perplexity(logits, y).item()
                    layer_norms_str = ', '.join([f'L{i}={v:.4f}' for i, v in enumerate(layer_norms)])
                    print(f"Step {current_step}: Loss={loss.item():.4f}, Perplexity={ppl:.4f}, GradNorm={grad_norm:.4f}, LR={cur_lr:.6f}, StepTime={step_time:.3f}s, Throughput={throughput:.0f} tok/s, ParamNorm={param_norm:.4f}, LayerNorms=[{layer_norms_str}]")
                    if args.use_wandb:
                        log_dict = {
                            "loss": loss.item(),
                            "perplexity": ppl,
                            "grad_norm": grad_norm,
                            "grad_clip_ratio": grad_clip_ratio,
                            "param_norm": param_norm,
                            "wall_clock_time": step_time,
                            "throughput": throughput,
                            "learning_rate": cur_lr,
                            "tokens_seen": (current_step + 1) * tokens_per_step,
                        }
                        for i, norm_val in enumerate(layer_norms):
                            log_dict[f"layer_{i}_activation_norm"] = norm_val
                        wandb.log(log_dict, step=current_step)

                if current_step % args.eval_num_iter == 0:
                    val_loss, val_ppl = evaluate(transformer, val_token_ids, args)
                    print(f"Step {current_step}: Val Loss={val_loss:.4f}, Val Perplexity={val_ppl:.4f}")
                    if args.use_wandb:
                        wandb.log({
                            "val_loss": val_loss,
                            "val_perplexity": val_ppl,
                        }, step=current_step)

                if current_step % args.generate_num_iter == 0:
                    text = generate(transformer, tokenizer, args.generate_prompt,
                                    max_length=args.generate_max_length, temperature=0.8, top_p=0.9, device='cuda')
                    print(f"Step {current_step} Generated: {text}")
                    if args.use_wandb:
                        wandb.log({"generated_text": wandb.Html(f"<pre>{text}</pre>")}, step=current_step)

                is_first_step = (current_step == 0)
                is_last_step = (current_step == max_iters - 1)
                is_periodic = (current_step % args.save_num_iter == 0)
                if is_first_step or is_last_step or is_periodic:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{current_step}.pt")
                    save_checkpoint(transformer, optimizer, current_step, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

                current_step += 1


if __name__ == "__main__":
    main()
