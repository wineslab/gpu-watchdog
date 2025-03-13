# Use an official TensorFlow image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Copy the script to check GPU utilization
COPY main.py /app/main.py

# Run the script to check GPU utilization
CMD ["python", "main.py"]