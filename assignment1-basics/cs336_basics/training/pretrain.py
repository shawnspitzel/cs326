import os
import os.path as osp
import wandb
import numpy as np
import torch
import cProfile
import pstats
from .loader import data_loading, load_checkpoint, save_checkpoint
from cs336_basics.model.optimizer import AdamW, SGDOptimizer
from cs336_basics.model.transformer import Transformer
from cs336_basics.model.loss import cross_entropy_loss, gradient_clipping, learning_rate_schedule
from cs336_basics.utils.args import get_args_pretrain


def get_checkpoint_dir(params):
    base_dir = "checkpoints"
    checkpoint_name = (
        f"nl{params['num_layers']}_"
        f"dm{params['d_model']}_"
        f"bs{params['batch_size']}_"
        f"lr{params['learning_rate']}_"
        f"seed{params['seed']}"
    )

    checkpoint_dir = osp.join(base_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir


def pretrain(model, train_data: np.array, val_data: np.array, optimizer, params, iteration: int):
    """
    Single training iteration with evaluation
    """

    model.train()
    inputs, targets = data_loading(train_data, params["batch_size"], params["context_length"], params["device"])
    optimizer.zero_grad()

    logits = model(inputs)
    loss = cross_entropy_loss(logits, targets)
    loss.backward()

    gradient_clipping(model.parameters(), max_norm=params['gradient_clip_norm'])

    lr = learning_rate_schedule(
        curr_iter=iteration,
        max_lr=params["learning_rate"],
        min_lr=params["min_lr"], 
        warm_iters=params["warmup_iters"],
        cos_iters=params["cosine_iters"]
        )
    optimizer.lr = lr
    optimizer.step()

    train_loss = loss.item()
    val_loss = None
    
    if iteration % params["eval_interval"] == 0:
        model.eval()
        val_losses = []

        with torch.no_grad():
            for _ in range(params["eval_iters"]):
                val_inputs, val_targets = data_loading(
                    val_data,
                    params["batch_size"],
                    params["context_length"],
                    params["device"]
                )
                val_logits = model(val_inputs)
                val_loss_batch = cross_entropy_loss(val_logits, val_targets)
                val_losses.append(val_loss_batch.item())
        val_loss = sum(val_losses) / len(val_losses)

    return train_loss, val_loss


def run(params):
    """
    Main training loop

    Inputs:
        params: dict of all hyperparameters from args/config
    """

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params["seed"])

    checkpoint_dir = get_checkpoint_dir(params)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    requested_device = params["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif requested_device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    for name, path in [("train_data", params["train_data"]), ("val_data", params["val_data"])]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"{name} file is empty: {path}")

    train_data = np.memmap(params["train_data"], dtype=params["data_dtype"], mode='r')
    val_data = np.memmap(params["val_data"], dtype=params["data_dtype"], mode='r')
    
    model = Transformer(
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            d_ff=params['d_ff'],
            num_layers=params['num_layers'],
            vocab_size=params['vocab_size'],
            context_length=params['context_length'],
            theta=params['theta']
        )

    model = model.to(device)
    if params["compile"]:
        model = torch.compile(model)
        
    if params["optimizer"] == "sgd":
        optimizer = SGDOptimizer(params=model.parameters(), lr=params["learning_rate"])
    else:
        optimizer = AdamW(
            params=model.parameters(), 
            betas=(params["beta1"], params["beta2"]),
            weight_decay=params["weight_decay"],
            lr=params["learning_rate"]
            )

    start_iter = 0
    if params["resume_from"]:
        resume_path = params["resume_from"]
        start_iter = load_checkpoint(resume_path, model, optimizer=optimizer, device=device)
    for iter in range(start_iter, params["max_iters"]):
        train_loss, val_loss = pretrain(
            model=model, 
            train_data=train_data, 
            val_data=val_data,
            optimizer=optimizer,
            params=params,
            iteration=iter
            )
        if iter % params["log_interval"] == 0 and wandb.run is not None:
            wandb.log({"train/loss": train_loss, "lr": optimizer.lr, "iteration": iter})

        if iter % params["eval_interval"] == 0 and val_loss is not None and wandb.run is not None:
            wandb.log({"val/loss": val_loss, "iteration": iter})

        if iter > 0 and iter % params["checkpoint_interval"] == 0:
            checkpoint_path = osp.join(checkpoint_dir, f"checkpoint_iter_{iter}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=iter,
                out=checkpoint_path
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    output_path = osp.join(checkpoint_dir, f"checkpoint_iter_{params['max_iters']}.pt")
    if params["max_iters"] % params["checkpoint_interval"] != 0:
        save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=params["max_iters"],
                out=output_path
            )
    print(f"Model saved to {output_path}")
    wandb.finish()

if __name__ == "__main__":
    params = get_args_pretrain()
    wandb.init(
        project="AtlasLM-Pretrain",
        name=f"Pretrain_layers_{params['num_layers']}_bs{params['batch_size']}_lr{params['learning_rate']}",
        mode="online",
        config=params,
    )

    config = wandb.config
    params.update(dict(config))

    if params.get("profile", False):
        profiler = cProfile.Profile()
        profiler.enable()

        run(params)

        profiler.disable()
        profile_output = params["profile_output"]

        os.makedirs(osp.dirname(profile_output), exist_ok=True)
        profiler.dump_stats(profile_output)
        
        print(f"\nProfile stats saved to: {profile_output}")

        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(20)
    else:
        run(params)