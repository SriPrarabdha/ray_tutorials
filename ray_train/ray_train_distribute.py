# ray_train_checkpoint.py
import os, torch, random, time
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import ray
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

def train_loop(config):
    # Simple synthetic data
    x = torch.randn(1000, 10)
    y = torch.randint(0, 2, (1000,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

    # Model + optimizer
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2)).cuda()
    opt = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0

    # Check if resuming from checkpoint
    ckpt = train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as ckpt_dir:
            checkpoint_state = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"))
            model.load_state_dict(checkpoint_state["model"])
            opt.load_state_dict(checkpoint_state["optimizer"])
            start_epoch = checkpoint_state["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, config["epochs"]):
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Randomly simulate worker failure to demo fault-tolerance
        if random.random() < 0.1:
            raise RuntimeError("Simulated worker failure!")

        # Save checkpoint every epoch
        with train.checkpoint_dir(step=epoch) as ckpt_dir:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(ckpt_dir, "checkpoint.pt"),
            )
        train.report({"epoch": epoch, "loss": loss.item()})

# Start Ray
ray.init(address="auto")  # or locally

trainer = TorchTrainer(
    train_loop_per_worker=train_loop,
    scaling_config=ScalingConfig(num_workers=4, 
                                 resources_per_worker={"CPU": 4, "GPU": 1, "GPU_TYPE": "A100"},),
    run_config=RunConfig(
        name="fault_tolerance_demo",
        storage_path="./ray_results",  # where checkpoints will be saved
        checkpoint_config=train.CheckpointConfig(num_to_keep=2),
        failure_config=train.FailureConfig(max_failures=3),  # auto-restart attempts
    ),
    train_loop_config={"lr": 1e-3, "epochs": 10},
)

result = trainer.fit()
print(result)
