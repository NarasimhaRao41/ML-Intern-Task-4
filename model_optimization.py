import tensorflow as tf
import numpy as np
import time
import os
import psutil

def get_file_size(file_path):
    
    return os.path.getsize(file_path) / (1024 * 1024)

def run_optimization_task():
    print("Starting ML Intern Task 4 ")
    
   
    print("\n Loading Baseline EfficientNetB0...")
    model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=True)
    
   
    if not os.path.exists('output'):
        os.makedirs('output')

    
    baseline_path = 'output/baseline_model.h5'
    model.save(baseline_path)
    baseline_size = get_file_size(baseline_path)
    
    
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

 
    print("\n Converting to TFLite with FP16 Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16] # FP16
    tflite_model = converter.convert()

    tflite_path = 'output/optimized_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    opt_size = get_file_size(tflite_path)
    print(f"      Optimized Size: {opt_size:.2f} MB")

    print("\n Measuring Optimized Speed...")
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

    print("\n[4/4] Saving Metrics Files...")
    with open('original_metrics.txt', 'w') as f:
        f.write(f"Model Size: {baseline_size:.2f} MB\nInference Time: {baseline_time:.2f} ms\n")
        
    with open('optimized_metrics.txt', 'w') as f:
        f.write(f"Model Size: {opt_size:.2f} MB\nInference Time: {opt_time:.2f} ms\n")
        
    print("\n Task Complete. Files ready in folder. ")

if __name__ == "__main__":
    run_optimization_task()
