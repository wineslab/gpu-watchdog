# Use an official PyTorch image with GPU support
FROM pytorch/pytorch:latest-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the script to check GPU utilization
COPY check_gpu_pt.py /app/check_gpu_pt.py

# Install any necessary dependencies (if needed)
# RUN pip install --no-cache-dir some-package

# Run the script to check GPU utilization
CMD ["python", "check_gpu_pt.py"]