import torch

a = torch.ones([2, 4], requires_grad=True)
aa = torch.tensor([[1, 1, 1, 0.9], [1, 1.1, 1, 0.9]], requires_grad=True)

func = torch.nn.MSELoss()

b = func(a, aa)

b.backward()
print(b)
print(a.grad)
