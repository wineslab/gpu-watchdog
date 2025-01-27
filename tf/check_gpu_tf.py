import tensorflow as tf

# Check if TensorFlow can see the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU is available and being utilized by TensorFlow.")
    for gpu in gpus:
        print(f"GPU: {gpu}")
else:
    print("No GPU found. TensorFlow is using the CPU.")