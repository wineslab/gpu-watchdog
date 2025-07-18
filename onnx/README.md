# ONNX Runtime GPU Testing

This folder contains a Docker setup to test ONNX Runtime with GPU acceleration.

## What it does

The test script (`main.py`) performs the following:

1. **Provider Detection**: Checks if CUDA execution provider is available in ONNX Runtime
2. **Model Creation**: Creates a simple ONNX model for matrix multiplication
3. **GPU Stress Test**: Runs matrix multiplication operations on the GPU (if available)
4. **Memory Monitoring**: Attempts to monitor GPU memory usage during operations
5. **Performance Logging**: Logs execution times and memory usage for each iteration

## Key Features

- Uses ONNX Runtime GPU (`onnxruntime-gpu`) package
- Fallback to CPU execution if GPU is not available
- Creates ONNX model programmatically for testing
- Monitors GPU memory usage when possible
- Comprehensive logging of execution details

## Running the Test

From the project root directory:

```bash
# Build and run the ONNX Runtime container
docker-compose up onnx

# Or build and run specifically
docker-compose build onnx
docker-compose run onnx
```

## Expected Output

When GPU is available:
- Detection of CUDAExecutionProvider
- Matrix multiplication performed on GPU
- GPU memory usage statistics
- Execution time measurements

When GPU is not available:
- Falls back to CPUExecutionProvider
- CPU-based matrix multiplication
- CPU execution time measurements

## Dependencies

- ONNX Runtime GPU
- NumPy
- CUDA runtime (provided by base image)
- Optional: pynvml for detailed GPU memory monitoring
