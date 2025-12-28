import torch
import numpy as np

print("Hello from bci2020 environment!")
print(f"PyTorch version: {torch.__version__}")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Create a random tensor of size (3, 3)
random_tensor = torch.rand(3, 3)
print("Random tensor:")
print(random_tensor)