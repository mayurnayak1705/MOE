import torch

x = torch.tensor([-0.0188, -0.3696, -0.1572, -0.2070,  0.3578, -0.4086,  0.0490,0.2428])
print(torch.nn.functional.softplus(x))