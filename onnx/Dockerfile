FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install ONNX Runtime GPU
RUN pip3 install --upgrade pip && \
    pip3 install onnxruntime-gpu onnx numpy

# Set the working directory
WORKDIR /app

# Copy the script to check GPU utilization
COPY main.py /app/main.py

# Run the script to check GPU utilization
CMD ["python3", "main.py"]
