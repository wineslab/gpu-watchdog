import onnxruntime as ort
import numpy as np
import time
import logging

# Configure logging with a detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Check available providers
available_providers = ort.get_available_providers()
logging.info(f"Available ONNX Runtime providers: {available_providers}")

# Check if CUDA provider is available
if 'CUDAExecutionProvider' in available_providers:
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    device_type = "GPU"
    logging.info("CUDA provider is available. Using GPU for inference.")
else:
    providers = ['CPUExecutionProvider']
    device_type = "CPU"
    logging.info("CUDA provider not available. Using CPU for inference.")

# Create a simple ONNX model for matrix multiplication
def create_matrix_multiplication_model():
    """Create a simple ONNX model that performs matrix multiplication"""
    import onnx
    from onnx import helper, TensorProto
    
    # Define input and output tensors
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, ['M', 'K'])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, ['K', 'N'])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, ['M', 'N'])
    
    # Create MatMul node
    matmul_node = helper.make_node('MatMul', ['A', 'B'], ['C'])
    
    # Create graph
    graph = helper.make_graph([matmul_node], 'matmul_graph', [A, B], [C])
    
    # Create model with compatible IR version
    model = helper.make_model(graph)
    model.ir_version = 10  # Set to maximum supported IR version
    model.opset_import[0].version = 17  # Use a stable opset version
    
    # Check the model
    onnx.checker.check_model(model)
    return model

try:
    # Create the ONNX model
    onnx_model = create_matrix_multiplication_model()
    
    # Serialize the model to bytes
    model_bytes = onnx_model.SerializeToString()
    
    # Create inference session
    session = ort.InferenceSession(model_bytes, providers=providers)
    
    logging.info(f"ONNX Runtime session created with providers: {session.get_providers()}")
    
    # Set matrix size for stress testing
    size = 2000  # Smaller size than torch/tf due to ONNX overhead
    logging.info(f"Matrix size set to: {size}x{size}")
    
    # Create large random matrices
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    logging.info("Starting ONNX Runtime GPU stress test...")
    
    # Get device information
    if 'CUDAExecutionProvider' in session.get_providers():
        try:
            # Try to get CUDA device info (this may not always work)
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            gpu_model = gpu_name
        except:
            gpu_model = "Unknown CUDA GPU"
    else:
        gpu_model = "N/A"
    
    # Run inference loop
    for i in range(10):
        start_time = time.time()
        try:
            # Run inference
            result = session.run(['C'], {'A': A, 'B': B})
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if 'CUDAExecutionProvider' in session.get_providers():
                try:
                    # Try to get memory info if pynvml is available
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used_memory_mb = mem_info.used / (1024 ** 2)
                    total_memory_mb = mem_info.total / (1024 ** 2)
                    logging.info(f"Iteration {i} on GPU (Model: {gpu_model}): "
                                f"Time {elapsed_time:.2f}s, GPU Memory Used: {used_memory_mb:.2f}/{total_memory_mb:.2f} MB")
                except:
                    logging.info(f"Iteration {i} on GPU (Model: {gpu_model}): "
                                f"Time {elapsed_time:.2f}s, Memory info unavailable")
            else:
                logging.info(f"Iteration {i} on CPU: Completed in {elapsed_time:.2f}s")
                
        except Exception as e:
            logging.error(f"Iteration {i}: Exception encountered: {e}")
            continue
            
except ImportError:
    logging.warning("ONNX package not available, creating a simplified test")
    
    # Fallback: Simple matrix multiplication without creating ONNX model
    # Create test matrices for basic provider testing
    size = 1000
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    logging.info("Running simplified ONNX Runtime provider test...")
    
    for i in range(5):
        start_time = time.time()
        # Simple numpy operation to test the environment
        C = np.matmul(A, B)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logging.info(f"Iteration {i}: NumPy matrix multiplication took {elapsed_time:.2f}s")
        logging.info(f"ONNX Runtime providers available: {available_providers}")

except Exception as e:
    logging.error(f"Error creating ONNX model or session: {e}")
    logging.info("Testing ONNX Runtime provider availability only...")
    
    # Just log the available providers
    for i in range(3):
        logging.info(f"Test {i}: ONNX Runtime providers: {available_providers}")
        if 'CUDAExecutionProvider' in available_providers:
            logging.info(f"Test {i}: GPU support detected in ONNX Runtime")
        else:
            logging.info(f"Test {i}: Only CPU support available in ONNX Runtime")
        time.sleep(1)
