import torch
import time
import logging
import argparse

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# If CUDA is available, log GPU properties
if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(device)
    logging.info(f"GPU Name: {gpu_props.name}")
    total_memory_mb = gpu_props.total_memory / (1024 ** 2)
    logging.info(f"Total GPU Memory: {total_memory_mb:.2f} MB")

# Set default tensor size
size = 10000

# Create large random matrices on the selected device
A = torch.randn(size, size, device=device)
B = torch.randn(size, size, device=device)

logging.info("Starting GPU stress test...")

iteration = 0
gpu_model = gpu_props.name if torch.cuda.is_available() else "N/A"
for i in range(100):
    start_time = time.time()
    try:
        # Perform matrix multiplication (GPU-intensive)
        C = torch.mm(A, B)
        torch.cuda.synchronize()  # Ensure all operations complete
    except Exception as e:
        logging.error(f"Iteration {i}: Exception encountered: {e}")
        continue

    elapsed_time = time.time() - start_time
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)

    logging.info(f"Iteration {i} on {device} (GPU Model: {gpu_model}): Matrix multiplication took {elapsed_time:.2f} seconds")
    logging.info(f"Iteration {i} on {device} (GPU Model: {gpu_model}): Time {elapsed_time:.2f}s, Allocated {allocated_memory:.2f} MB, Reserved {reserved_memory:.2f} MB")
