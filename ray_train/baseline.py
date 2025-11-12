# train_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Dummy data
x = torch.randn(5000, 10)
y = torch.randint(0, 2, (5000,))
loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

# Simple model
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    for xb, yb in loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")


# What if we have very large data ?
# What if i want distributed training ?
# What if i distributed it is it actually fault tolerant ?