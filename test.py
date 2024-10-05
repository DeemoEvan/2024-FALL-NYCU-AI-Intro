import torch
import torch.nn as nn
import numpy as np
data = [[1, 2], [3, 4]]
x_data = np.array(data)
x_data = torch.tensor(data)
t1 = torch.ones(2,4)
print(t1)
t2 = torch.zeros(2,1)
print(t2)
t3 = torch.cat((t1, t2), 1)
print(t3)

model = nn.Sequential(
  nn.Linear(3,5),
  nn.ReLU(),
  nn.Linear(5,1),
  nn.ReLU(),
  nn.Linear(5,5)
)



