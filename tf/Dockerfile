# Use an official TensorFlow image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Copy the script to check GPU utilization
COPY check_gpu_tf.py /app/check_gpu_tf.py

# Install any necessary dependencies (if needed)
# RUN pip install --no-cache-dir some-package

# Run the script to check GPU utilization
CMD ["python", "check_gpu_tf.py"]