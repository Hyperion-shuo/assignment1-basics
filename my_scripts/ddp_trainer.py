import os
import time
import numpy.typing as npt
import torch
import torch.cuda.nvtx as nvtx
import wandb
import argparse
import numpy as np
from contextlib import nullcontext

from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch import linalg as LA
import torch.distributed as dist
import torch.multiprocessing as mp
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

from cs336_basics.transformer.transformer import TransformerLM
from cs336_basics.tokenizer import TrainedTokenizer
from cs336_basics.train.data_loader import dataloader, dataloader_sequential, get_tokenized_data_files
from cs336_basics.train.optimizer import AdamW, lr_cos_schedule, gradient_clipping
from cs336_basics.train.loss import cross_entropy_loss, perplexity
from cs336_basics.train.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.generate import generate
from cs336_basics.collective_communication_utils import setup, cleanup, ddp_wrapper, ddp_bucket_wrapper

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    parser.add_argument("--save_num_iter", type=int, default=5000)
    parser.add_argument("--log_num_iter", type=int, default=100)
    parser.add_argument("--generate_num_iter", type=int, default=2000)
    parser.add_argument("--generate_prompt", type=str, default="Once upon a time")
    parser.add_argument("--generate_max_length", type=int, default=100)
    parser.add_argument("--eval_num_iter", type=int, default=500)
    parser.add_argument("--eval_num_batches", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=10000, help="Maximum number of training iterations. If num_epoch implies more iters, cap at max_iter.")
    parser.add_argument("--use_mixed_precision", action="store_true", default=False,
                        help="Enable mixed precision training with bfloat16.")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable wandb logging")
    parser.add_argument("--wandb_project_name", type=str, default=None, help="Auto-inferred if not set.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Auto-inferred if not set.")

    # hyper params
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Per-rank (local) batch size. Global batch size = batch_size * world_size.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--alpha_min", type=float, default=1e-4)
    parser.add_argument("--alpha_max", type=float, default=1e-3)
    parser.add_argument("--t_w", type=int, default=1000)
    parser.add_argument("--t_c", type=int, default=10000)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999], help="AdamW betas, e.g. 0.9 0.95")
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Auto-inferred if not set.")
    parser.add_argument("--checkpoint_load_path", type=str, default=None, 
                        help="Auto-inferred if not set. Set to '' to disable loading.")
    
    # ddp
    parser.add_argument("--ddp_backend", type=str, default="nccl", choices=["nccl", "gloo"], 
                        help="DDP backend to use. Default 'nccl' for GPU training.")
    parser.add_argument("--world_size", type=int, default=4, help="Number of processes for DDP.")
    parser.add_argument("--flatten_dense_tensors", action="store_true", default=False, 
                        help="Whether to flatten dense tensors for communication efficiency.")
    parser.add_argument("--use_bucket_comm", action="store_true", default=False,
                        help="Use bucket-based gradient communication with async all-reduce.")
    parser.add_argument("--bucket_size_mb", type=float, default=25.0,
                        help="Bucket size in MB for bucket-based gradient communication.")
    parser.add_argument("--use_overlap_comm", action="store_true", default=False,
                        help="Use ddp_wrapper to overlap communication and computation via per-param async all-reduce hooks.")
    parser.add_argument("--use_nvtx", action="store_true", default=False,
                        help="Enable NVTX range annotations for Nsight Systems profiling.")

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
        args.wandb_project_name = f"{args.data_set_name}-ddp"
    if args.wandb_run_name is None:
        args.wandb_run_name = f"ddp_w{args.world_size}_d{args.d_model}_h{args.num_heads}_l{args.num_layers}_ff{args.d_ff}_b{args.batch_size}"
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


def dataloader_sequential_with_rank(token_ids, B, T, device, rank, world_size):
    """Sequential dataloader that yields non-overlapping batches for each rank.
    
    Each rank processes a different slice of each global batch.
    Global batch size = B * world_size. Each rank gets B samples.
    The data is partitioned so that rank i gets rows [i*B, (i+1)*B) of the global batch.
    """
    global_B = B * world_size
    num_samples = len(token_ids)
    needed_tokens = global_B * T + 1
    current_idx = 0

    while current_idx + needed_tokens <= num_samples:
        # Load the full global batch worth of tokens
        global_batch = torch.from_numpy(token_ids[current_idx:current_idx + needed_tokens]).to(device)
        current_idx += needed_tokens - 1

        # Reshape into global_B sequences of length T (and T+1 for targets)
        # x: global_batch[:global_B*T] reshaped to (global_B, T)
        # y: global_batch[1:global_B*T+1] reshaped to (global_B, T)
        x_all = global_batch[:global_B * T].reshape(global_B, T)
        y_all = global_batch[1:global_B * T + 1].reshape(global_B, T)

        # Each rank takes its own slice
        start = rank * B
        end = (rank + 1) * B
        yield x_all[start:end], y_all[start:end]


@torch.no_grad()
def evaluate(model, val_token_ids, args, device):
    """Evaluate model on validation set, return avg loss and perplexity."""
    model.eval()
    val_loader = dataloader(
        token_ids=val_token_ids,
        B=args.batch_size,
        T=args.context_length,
        device=device,
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


def main(rank, world_size):
    setup(rank, world_size, backend='nccl')
    
    args = parse_args()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    is_master = (rank == 0)

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
    ).to(device)

    if args.use_bucket_comm:
        # ddp_bucket_wrapper handles broadcast internally
        transformer = ddp_bucket_wrapper(transformer, bucket_size_mb=args.bucket_size_mb)
        optimizer = AdamW(transformer.module.parameters())
    elif args.use_overlap_comm:
        # ddp_wrapper handles broadcast internally, registers per-param async all-reduce hooks
        transformer = ddp_wrapper(transformer)
        optimizer = AdamW(transformer.module.parameters())
    else:
        # Broadcast model parameters from rank 0 to ensure all ranks start with the same weights
        for param in transformer.parameters():
            dist.broadcast(param.data, src=0)
        optimizer = AdamW(transformer.parameters())
    # if args.flatten_dense_tensors:
    #     grads_list = [p.grad for p in transformer.parameters() if p.grad is not None]
    #     flat_grads, grad_shapes = None, None

    current_step = 0

    # load val data (only needed on rank 0 for evaluation)
    val_token_ids = None
    if is_master:
        val_token_path = os.path.join(base_dir, 'data', f'{args.data_set_name}_val_tokens.npy')
        val_token_ids = np.load(val_token_path)
        print(f"Loaded val tokens: {len(val_token_ids)}")

    if args.use_wandb and is_master:
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args))

    # compute max iters from total_tokens
    # Global batch size = batch_size * world_size, so tokens_per_iter accounts for all ranks
    tokens_per_iter = args.batch_size * args.context_length * world_size
    max_iters_from_tokens = args.total_tokens // tokens_per_iter
    max_iters = min(max_iters_from_tokens, args.max_iter)
    if is_master:
        print(f"Training for {max_iters} iters (from_tokens={max_iters_from_tokens}, max_iter={args.max_iter}, tokens_per_iter={tokens_per_iter}, world_size={world_size})")

    # Mixed precision setup (bf16 doesn't need GradScaler)
    if args.use_mixed_precision:
        autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if is_master:
            print("Mixed precision training enabled (bfloat16)")
    else:
        autocast_ctx = nullcontext()

    done = False
    for _ in range(args.num_epoch):
        if done:
            break
        tokenized_data_files = get_tokenized_data_files(os.path.join(base_dir, 'data'), args.data_set_name)
        for tokenized_data_file in tokenized_data_files:
            if done:
                break
            with open(tokenized_data_file, 'rb') as f:
                token_ids = np.load(f)['tokens']
            
            # Each rank gets non-overlapping slices of the global batch
            train_dataloader = dataloader_sequential_with_rank(
                token_ids=token_ids,
                B=args.batch_size,
                T=args.context_length,
                device=device,
                rank=rank,
                world_size=world_size,
            )

            for x, y in train_dataloader:
                if current_step >= max_iters:
                    done = True
                    break

                # Start NVTX profiling after warmup
                if args.use_nvtx and current_step == 10:
                    torch.cuda.cudart().cudaProfilerStart()

                step_start_time = time.time()
                nvtx.range_push(f"step_{current_step}") if args.use_nvtx else None
                transformer.train()
                x, y = x.long(), y.long()

                is_log_step = is_master and (current_step % args.log_num_iter == 0)
                with nvtx.range("forward") if args.use_nvtx else nullcontext():
                    with autocast_ctx:
                        logits = transformer(x)

                with nvtx.range("loss") if args.use_nvtx else nullcontext():
                    with autocast_ctx:
                        loss = cross_entropy_loss(logits, y)

                cur_lr = lr_cos_schedule(current_step, args.alpha_min, args.alpha_max, args.t_w, args.t_c)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr

                optimizer.zero_grad()

                # For overlap/bucket: async all-reduce hooks fire during backward,
                # so backward and comm overlap in the timeline.
                with nvtx.range("backward") if args.use_nvtx else nullcontext():
                    loss.backward()

                # All-reduce gradients: sum and average across all ranks
                # Time the communication
                torch.cuda.synchronize()
                comm_start = time.time()
                if args.use_bucket_comm or args.use_overlap_comm:
                    # Bucket / overlap version: all-reduce is triggered by hooks during backward,
                    # just wait for all async ops to finish
                    with nvtx.range("comm_wait") if args.use_nvtx else nullcontext():
                        transformer.finish_gradient_synchronization()
                elif args.flatten_dense_tensors:
                    # Flatten all gradients into a single tensor for more efficient communication
                    with nvtx.range("comm_allreduce") if args.use_nvtx else nullcontext():
                        params_with_grad = [p for p in transformer.parameters() if p.grad is not None]
                        grads_list = [p.grad.data for p in params_with_grad]
                        flat_grads = torch._utils._flatten_dense_tensors(grads_list)
                        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
                        flat_grads /= world_size
                        unflattened_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads_list)
                        # Unflatten back into individual gradients
                        for p, unflat_g in zip(params_with_grad, unflattened_grads):
                            p.grad.copy_(unflat_g)
                else:
                    with nvtx.range("comm_allreduce") if args.use_nvtx else nullcontext():
                        for param in transformer.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                                param.grad.data /= world_size
                torch.cuda.synchronize()
                comm_time = time.time() - comm_start

                model_params = transformer.module.parameters() if (args.use_bucket_comm or args.use_overlap_comm) else transformer.parameters()
                grad_norm = get_grad_norm(model_params)
                gradient_clipping(model_params, args.max_grad_norm)
                with nvtx.range("optimizer_step") if args.use_nvtx else nullcontext():
                    optimizer.step()

                nvtx.range_pop() if args.use_nvtx else None

                # compute step timing and throughput
                torch.cuda.synchronize()
                step_time = time.time() - step_start_time
                tokens_per_step = args.batch_size * args.context_length * world_size  # global tokens per step
                throughput = tokens_per_step / step_time  # tokens/sec
                comm_fraction = comm_time / step_time  # fraction of step spent on communication

                # compute grad clip ratio and param norm
                grad_clip_ratio = grad_norm / args.max_grad_norm
                param_norm = get_param_norm(model_params)

                if is_log_step:
                    ppl = perplexity(logits, y).item()
                    print(f"Step {current_step}: Loss={loss.item():.4f}, Perplexity={ppl:.4f}, GradNorm={grad_norm:.4f}, LR={cur_lr:.6f}, StepTime={step_time:.3f}s, CommTime={comm_time:.3f}s ({comm_fraction*100:.1f}%), Throughput={throughput:.0f} tok/s, ParamNorm={param_norm:.4f}")
                    if args.use_wandb:
                        log_dict = {
                            "loss": loss.item(),
                            "perplexity": ppl,
                            "grad_norm": grad_norm,
                            "grad_clip_ratio": grad_clip_ratio,
                            "param_norm": param_norm,
                            "wall_clock_time": step_time,
                            "comm_time": comm_time,
                            "comm_fraction": comm_fraction,
                            "throughput": throughput,
                            "learning_rate": cur_lr,
                            "tokens_seen": (current_step + 1) * tokens_per_step,
                        }
                        wandb.log(log_dict, step=current_step)

                # Evaluation: only on rank 0
                if is_master and current_step % args.eval_num_iter == 0:
                    val_loss, val_ppl = evaluate(transformer, val_token_ids, args, device)
                    print(f"Step {current_step}: Val Loss={val_loss:.4f}, Val Perplexity={val_ppl:.4f}")
                    if args.use_wandb:
                        wandb.log({
                            "val_loss": val_loss,
                            "val_perplexity": val_ppl,
                        }, step=current_step)

                # Generation: only on rank 0
                if is_master and current_step % args.generate_num_iter == 0:
                    text = generate(transformer, tokenizer, args.generate_prompt,
                                    max_length=args.generate_max_length, temperature=0.8, top_p=0.9, device=device)
                    print(f"Step {current_step} Generated: {text}")
                    if args.use_wandb:
                        wandb.log({"generated_text": wandb.Html(f"<pre>{text}</pre>")}, step=current_step)

                # Checkpoint: only on rank 0
                is_first_step = (current_step == 0)
                is_last_step = (current_step == max_iters - 1)
                is_periodic = (current_step % args.save_num_iter == 0)
                if is_master and (is_first_step or is_last_step or is_periodic):
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{current_step}.pt")
                    save_checkpoint(transformer, optimizer, current_step, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

                current_step += 1

    # Ensure all ranks finish before cleanup
    dist.barrier()
    if args.use_wandb and is_master:
        wandb.finish()
    cleanup()

# nohup uv run python -m my_scripts.ddp_trainer --wandb_project_name tinystories-ablation --wandb_run_name ddp_baseline > nohup_scripts/logs/ddp_baseline.log 2>&1 &
# nohup uv run python -m my_scripts.ddp_trainer --wandb_project_name tinystories-ablation --flatten_dense_tensors --wandb_run_name ddp_flatten_dense_vector > nohup_scripts/logs/ddp_flatten_dense_vector.log 2>&1 &
if __name__ == "__main__":
    args = parse_args()
    WORLD_SIZE = args.world_size
    mp.spawn(main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
