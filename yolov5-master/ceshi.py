import numpy as np
import cv2
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_capability())

x = torch.ones((3, 3)).to(device)

# y = torch.ones((3, 3, 3), device=device)
print(x.device)

net = nn.Linear(3, 1).cuda()


y = net(x)
print(y)

for name, param in net.named_parameters():
    print(param, param.shape)