# Use an official PyTorch image with GPU support
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Copy the script to check GPU utilization
COPY main.py /app/main.py

# Run the script to check GPU utilization
CMD ["python", "main.py"]