# gpu-watchdog
GPU Watchdog is your go-to toolkit for monitoring, testing, and validating GPU utilization in deep learning frameworks like TensorFlow and PyTorch.

Ensure that you have NVIDIA Docker installed and configured on your system to use GPU resources within Docker containers.

The --gpus all flag is used to enable GPU access in the container.

The latest-gpu and latest-cuda11.7-cudnn8-runtime tags are used to ensure that the Docker images come with GPU support. You can adjust these tags based on your specific needs.