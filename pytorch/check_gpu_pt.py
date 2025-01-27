import torch

# Check if PyTorch can see the GPU
if torch.cuda.is_available():
    print("GPU is available and being utilized by PyTorch.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. PyTorch is using the CPU.")