import torch
import torch.nn as nn
import numpy as np
x = torch.ones(16, 28, 28)
W = torch.ones(28, 128)
out = x @ W
print(out.shape)



