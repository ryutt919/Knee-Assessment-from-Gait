import torch
import torch.nn as nn
import numpy as np
from ML_based.models.transformer import TransformerNet

net = TransformerNet(9, 64, 4, 2, 128, 0.3, 0.1, 2, 0).to("cpu")
net.train()
x = torch.randn(16, 101, 9)
mask = torch.zeros(16, 101, dtype=torch.bool)
mask[:, -10:] = True
print("Forward...")
out = net(x, None, mask)
print("Backward...")
out.mean().backward()
print("Done!")
