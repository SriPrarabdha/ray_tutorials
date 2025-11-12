# ray_llama_factory_train.py
import os
import ray
import ray.data as rd
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

# Imports from LLaMA Factory (installed via pip install llama-factory)
from llmtuner import run_exp, prepare_args

def train_loop_per_worker(config):
    """Each Ray worker runs a fine-tuning job using LLaMA Factory Trainer."""
    import torch
    from ray import train

    # Get dataset shard
    ds_shard = train.get_dataset_shard("train")
    # Convert to an iterable of text samples
    iterator = ds_shard.iter_batches(batch_size=config["batch_size"], batch_format="pandas")

    # Prepare LLaMA Factory arguments dynamically
    model_args = {
        "model_name_or_path": config["model_name"],
        "output_dir": "./output/checkpoints",
        "train_on_inputs": False,
        "use_peft": True,
        "lora_r": 8,
        "lora_alpha": 32,
        "bf16": True,
        "learning_rate": config["lr"],
        "num_train_epochs": config["epochs"],
        "per_device_train_batch_size": config["batch_size"],
        "gradient_accumulation_steps": 4,
        "save_steps": 500,
        "logging_steps": 50,
    }

    args = prepare_args(model_args)
    trainer = run_exp(args, resume=False, train_data_iterator=iterator)

    trainer.train()
    trainer.save_model()
    train.report({"epoch": config["epochs"], "status": "completed"})

# --- Entry point ---
if __name__ == "__main__":
    ray.init(address="auto")

    # Example: large dataset from Hugging Face or your own
    ds = rd.read_parquet("s3://mybucket/finetune_parquet/")
    ds = ds.repartition(16)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
        run_config=RunConfig(
            name="llama_factory_ray_finetune",
            storage_path="./ray_results",
        ),
        train_loop_config={
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "lr": 2e-5,
            "batch_size": 2,
            "epochs": 3,
        },
        datasets={"train": ds},
    )

    result = trainer.fit()
    print("Training completed:", result)
