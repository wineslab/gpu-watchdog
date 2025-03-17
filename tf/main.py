import tensorflow as tf
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if GPUs are available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logging.info("GPU is available and being utilized by TensorFlow.")
    for gpu in gpus:
        logging.info(f"GPU: {gpu}")
    # Retrieve the model name for the first GPU
    gpu_details = tf.config.experimental.get_device_details(gpus[0])
    gpu_model = gpu_details.get('device_name', 'Unknown')
    device_name = "/GPU:0"
else:
    logging.info("No GPU found. TensorFlow is using the CPU.")
    gpu_model = "N/A"
    device_name = "CPU"

# Set default matrix size
size = 10000
logging.info(f"Matrix size set to: {size}")

# Create large random matrices
A = tf.random.normal((size, size))
B = tf.random.normal((size, size))

logging.info("Starting GPU stress test...")

iteration = 0
for i in range(10):
    start_time = time.time()
    try:
        # Perform matrix multiplication (GPU-intensive)
        C = tf.matmul(A, B)
    except Exception as e:
        logging.error(f"Iteration {i}: An error occurred: {e}")
        continue

    end_time = time.time()
    elapsed_time = end_time - start_time

    # If running on GPU, attempt to log memory stats
    if gpus:
        try:
            mem_info = tf.config.experimental.get_memory_info(device_name)
            current_mem = mem_info.get('current', 0) / (1024 ** 2)
            peak_mem = mem_info.get('peak', 0) / (1024 ** 2)
            logging.info(f"Iteration {i} on GPU (Model: {gpu_model}): "
                         f"Time {elapsed_time:.2f}s, Current Memory: {current_mem:.2f} MB, Peak Memory: {peak_mem:.2f} MB")
        except Exception as e:
            logging.error(f"Iteration {i} on GPU (Model: {gpu_model}): "
                          f"Could not retrieve memory info: {e}")
    else:
        logging.info(f"Iteration {i} on CPU: Completed in {elapsed_time:.2f}s")