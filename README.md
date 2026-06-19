# Hardware-Aware AI Optimization for In-Vehicle Real-Time Applications 🚀

This repository contains the official implementation of the optimization pipeline developed to deploy high-fidelity perception models (RF-DETR) on resource-constrained automotive edge hardware (NVIDIA Jetson AGX Orin).

By bridging the gap between theoretical Frugal AI frameworks and low-level TensorRT compiler constraints, this project achieves a **133.6% throughput acceleration (141+ FPS)** for real-time person-segmentation and anonymization.

## ✨ Key Architectural Innovations
* **Native QDQ Bridge:** A custom engineering solution that forces PyTorch into strict symmetric quantization compliance for NVIDIA TensorRT, completely eliminating the "Fake Quantization Tax."
* **Algorithmic Stabilization:** Implements a "Smart Gradient Shield" and Delayed Pruning Strategy to prevent Neural Capacity Collapse during extreme INT8/INT4 Quantization-Aware Training (QAT).
* **Mixed-Precision Routing:** Strategic INT8 backbone execution with an FP16 fallback for sensitive Attention mechanisms, dropping latency to **7.05 ms**.

## 📂 Repository Structure
* `only_train_once/`: The core GETA joint-optimization framework.
* `rf_detr/`: The modified Vision Transformer architecture.
* `scripts/`: Contains the end-to-end pipeline.
  * `train.py`: The stabilized joint pruning & QAT training loop.
  * `patches.py` & `qdq_layers.py`: Custom runtime monkey patches and symmetric quantization layers.
  * `prune.py` & `export.py`: Deep graph surgery and ONNX export for TensorRT.

## 📊 Hardware Validation (NVIDIA Jetson AGX Orin)

| Metric | FP16 Baseline | Optimized (INT8 GETA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | 60.68 FPS | 141.79 FPS | **+133.6%** |
| **Latency** | 16.48 ms | 7.05 ms | **-57.2%** |
| **Energy/Frame**| 0.594 J | 0.234 J | **-60.6%** |

---
## 🚀 Quick Start
```bash
git clone https://github.com/Shaima-Alshamiry/hardware-aware-rfdetr-geta.git
cd hardware-aware-rfdetr
pip install --upgrade pip
pip install -r requirements.txt

## 📜 Acknowledgements & GETA Framework
This research heavily utilizes and builds upon the **GETA** (Generic, Efficient Training Architecture) framework for automated joint structured pruning and mixed-precision quantization. 

If you find the base optimization framework useful, please cite the original authors:
```bibtex
@article{qu2025automatic,
  title={Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression},
  author={Qu, Xiaoyi and Aponte, David and Banbury, Colby and Robinson, Daniel P and Ding, Tianyu and Koishida, Kazuhito and Zharkov, Ilya and Chen, Tianyi},
  journal={arXiv preprint arXiv:2502.16638},
  year={2025}
}
