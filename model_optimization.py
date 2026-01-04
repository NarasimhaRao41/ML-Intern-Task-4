import tensorflow as tf
import numpy as np
import time
import os
import psutil

def get_file_size(file_path):
    """Helper to get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def run_optimization_task():
    print("--- Starting ML Intern Task 4 ---")
    
    # 1. SETUP & BASELINE
    print("\n[1/4] Loading Baseline EfficientNetB0...")
    model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=True)
    
    # Create output directory
    if not os.path.exists('output'):
        os.makedirs('output')

    # Save Baseline to measure size
    baseline_path = 'output/baseline_model.h5'
    model.save(baseline_path)
    baseline_size = get_file_size(baseline_path)
    
    # Measure Baseline Speed
    print("      Measuring Baseline Speed (this takes a moment)...")
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    model.predict(input_data, verbose=0) # Warmup
    
    start_time = time.time()
    for _ in range(50):
        model.predict(input_data, verbose=0)
    end_time = time.time()
    baseline_time = ((end_time - start_time) / 50) * 1000
    
    print(f"      Baseline Size: {baseline_size:.2f} MB")
    print(f"      Baseline Time: {baseline_time:.2f} ms")

    # 2. OPTIMIZATION (TFLite + FP16)
    print("\n[2/4] Converting to TFLite with FP16 Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16] # FP16
    tflite_model = converter.convert()

    tflite_path = 'output/optimized_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    opt_size = get_file_size(tflite_path)
    print(f"      Optimized Size: {opt_size:.2f} MB")

    # 3. MEASURE OPTIMIZED SPEED
    print("\n[3/4] Measuring Optimized Speed...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke() # Warmup
    
    start_time = time.time()
    for _ in range(50):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    end_time = time.time()
    opt_time = ((end_time - start_time) / 50) * 1000
    
    print(f"      Optimized Time: {opt_time:.2f} ms")

    # 4. GENERATE METRICS FILES
    print("\n[4/4] Saving Metrics Files...")
    with open('original_metrics.txt', 'w') as f:
        f.write(f"Model Size: {baseline_size:.2f} MB\nInference Time: {baseline_time:.2f} ms\n")
        
    with open('optimized_metrics.txt', 'w') as f:
        f.write(f"Model Size: {opt_size:.2f} MB\nInference Time: {opt_time:.2f} ms\n")
        
    print("\n--- Task Complete. Files ready in folder. ---")

if __name__ == "__main__":
    run_optimization_task()
