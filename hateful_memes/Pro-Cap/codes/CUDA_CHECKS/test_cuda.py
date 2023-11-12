import torch

print(torch.cuda.is_available())
print()
print(torch.cuda.device_count())
print()
print(torch.cuda.get_device_name(0))
print()
print(torch.cuda.device(0))
print()
print(torch.cuda.current_device())
print()
print(torch.cuda.get_device_capability(0))