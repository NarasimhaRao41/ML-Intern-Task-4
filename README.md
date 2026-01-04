# ML Intern Task 4: EfficientNetB0 Model Optimization

This repository contains the optimization pipeline for the EfficientNetB0 deep learning model, targeting edge devices.

## 📂 Deliverables
- `model_optimization.py`: Script to load, measure, and optimize the model.
- `output/optimized_model.tflite`: The final optimized model (FP16 Quantized).
- `comparison_report.txt`: Detailed breakdown of performance gains.

## 🚀 Results Summary
| Metric | Original | Optimized | Improvement |
| :--- | :--- | :--- | :--- |
| **Size** | 21.03 MB | 10.14 MB | **~52% Smaller** |
| **Speed** | 164.06 ms | 45.28 ms | **~72% Faster** |

## 🛠️ How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow numpy psutil
