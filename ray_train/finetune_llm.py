# train_ray_finetune.py
"""
Distributed fine-tuning with Ray Train using PEFT (LoRA).
- Reads dataset via ray.data (Parquet shards on S3/local)
- Each worker gets a dataset shard via trainer datasets
- Uses Hugging Face Transformers + bitsandbytes + peft for efficient fine-tuning
- Checkpointing and failure-resilience via Ray Train APIs
"""

import os
import argparse
import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import ray.data as rd
import numpy as np

def preprocess_batch_for_torch(batch, tokenizer, max_length=1024):
    # batch is a dict of lists/arrays
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    # Convert lists to torch tensors with padding in collate_fn
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def train_loop_per_worker(config):
    """
    This runs on each worker. Use train.get_dataset_shard("train") to stream
    the training data for this worker.
    """
    import os
    import torch
    from torch.utils.data import DataLoader
    from ray import train
    from transformers import DataCollatorForLanguageModeling

    # configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    worker_rank = train.world_rank()
    world_size = train.world_size()
    print(f"Worker rank {worker_rank}/{world_size} starting on device {device}")

    # load tokenizer & model locally (each worker loads its own model/LoRA wrappers)
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # load pretrained model in 8-bit if configured to save memory (optional)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Apply PEFT LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
    )
    model = get_peft_model(model, peft_config)
    model.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # data: get the per-worker shard from Ray Train
    ds_shard = train.get_dataset_shard("train")
    # Use iter_batches to stream batches. batch_format="numpy" yields numpy arrays/dicts
    iterator = ds_shard.iter_batches(batch_size=config["batch_size"], batch_format="numpy")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

    epoch = 0
    ckpt = train.get_checkpoint()
    if ckpt:
        # restore model and optimizer if checkpoint exists
        with ckpt.as_directory() as ckpt_dir:
            state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            epoch = state.get("epoch", 0) + 1
        print(f"Resumed from checkpoint at epoch {epoch}")

    for e in range(epoch, config["epochs"]):
        for batch in iterator:
            # batch fields: "input_ids" (lists), "attention_mask" (lists) or pre-tokenized tensors
            # Convert to lists of strings if necessary — here we assume tokens are already present
            # Convert to tensors with collator
            try:
                torch_batch = {}
                # If dataset contains token lists, collate to tensors
                if isinstance(batch["input_ids"][0], (list, np.ndarray)):
                    # build a list of dicts for collator to process if necessary
                    # collator expects input: list[str] or dict of tensors; we can construct tensors directly
                    inputs = {"input_ids": [torch.tensor(x, dtype=torch.long) for x in batch["input_ids"]]}
                    # pad to max length within collator
                    batch_t = data_collator(inputs)
                else:
                    # If already numeric arrays
                    batch_t = data_collator({"input_ids": batch["input_ids"]})
            except Exception as ex:
                # Fallback: naive conversion
                input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(batch.get("attention_mask", np.ones_like(batch["input_ids"])), dtype=torch.long)
                batch_t = {"input_ids": input_ids, "attention_mask": attention_mask}

            # move to device
            input_ids = batch_t["input_ids"].to(device)
            attention_mask = batch_t.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save checkpoint at end of epoch
        with train.checkpoint_dir(step=e) as ckpt_dir:
            save_path = os.path.join(ckpt_dir, "model.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": e}, save_path)

        train.report({"epoch": e, "loss": loss.item()})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen-3")  # replace with exact HF id
    parser.add_argument("--data_path", type=str, default="./parquet_shards")  # local dir or s3://bucket/...
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # initialize Ray (auto-connect to cluster if running remotely)
    ray.init(address="auto")

    # Build Ray dataset from Parquet shards (supports s3:// paths)
    ds = rd.read_parquet(args.data_path)

    # Optionally repartition to control parallelism (e.g., num_workers * 4)
    ds = ds.repartition(args.num_workers * 4)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu),
        run_config=RunConfig(
            name="qwen-finetune-ray",
            storage_path="./ray_results/qwen_finetune",
            checkpoint_config=CheckpointConfig(num_to_keep=3),
        ),
        train_loop_config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        },
        datasets={"train": ds}
    )

    result = trainer.fit()
    print("Training finished. Result:", result)

if __name__ == "__main__":
    main()
