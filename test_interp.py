import torch.nn.functional as F
import torch


a = torch.tensor([[1, 1], [1, 1]])
a_pad = F.pad(input=a, pad=(1, 1, 1, 1), mode='constant', value=0)
print(a)
print(a_pad)