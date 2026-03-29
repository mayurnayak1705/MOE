import torch

x = torch.tensor([-0.0188, -0.3696, -0.1572, -0.2070,  0.3578, -0.4086,  0.0490,0.2428])
# print(torch.nn.functional.softplus(x))
y = torch.tensor([[2, 3],
        [3, 7],
        [7, 4],
        [7, 9],
        [7, 6],
        [7, 3],
        [7, 3],
        [3, 9],
        [3, 1],
        [0, 7]])
mask = (y==7).any(dim=-1)
m = mask.view(-1)
v = torch.rand(20,32)
y = torch.tensor([False, False, False, False,  True, False, False, False, False,  True,
         True,  True, False,  True,  True,  True, False, False, False, False])


print(v[y].shape)