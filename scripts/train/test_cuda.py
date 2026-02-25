import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


# lerobot 0.3.4 requires torch<2.8.0,>=2.2.1, but you have torch 2.11.0.dev20260201+cu128 which is incompatible.
# lerobot 0.3.4 requires torchvision<0.23.0,>=0.21.0, but you have torchvision 0.25.0.dev20260201+cu128 which is incompatible.
