# Hardware-Aware AI Optimization for In-Vehicle Real-Time Applications 🚀

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.x-green.svg)](https://developer.nvidia.com/tensorrt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the optimization pipeline developed for the Erasmus Mundus IPCVAI Master's Thesis. We bridge the gap between theoretical **Frugal AI** frameworks and low-level TensorRT compiler constraints to deploy high-fidelity **RF-DETR** perception models on resource-constrained automotive edge hardware (**NVIDIA Jetson AGX Orin**).

By bridging these layers, this project achieves a **133.6% throughput acceleration (141+ FPS)** for real-time person-segmentation and anonymization.

---

## 🛠 Architectural Overview
We implemented a decoupled generative paradigm to ensure GDPR compliance while maximizing throughput:

![Decoupled Generative Paradigm](assets/privacy_paradigm.png)

### Key Engineering Interventions
* **Native QDQ Bridge:** A custom engineering solution that forces PyTorch into strict symmetric quantization compliance for NVIDIA TensorRT, completely eliminating the "Fake Quantization Tax."
* **Algorithmic Stabilization:** Implements a "Smart Gradient Shield" and Delayed Pruning Strategy to prevent Neural **Capacity Collapse** during extreme INT8/INT4 Quantization-Aware Training (QAT).
* **Mixed-Precision Routing:** Strategic INT8 backbone execution with an FP16 fallback for sensitive Attention mechanisms, reducing latency to **7.05 ms**.

---

## 📂 Repository Structure
* `only_train_once/`: The core GETA joint-optimization framework.
* `rf_detr/`: The modified Vision Transformer architecture.
* `scripts/`: Contains the end-to-end pipeline:
  * `train.py`: The stabilized joint pruning & QAT training loop.
  * `patches.py` & `qdq_layers.py`: Custom runtime monkey patches and symmetric quantization layers.
  * `prune.py` & `export.py`: Deep graph surgery and ONNX export for TensorRT.

---

## 📊 Hardware Validation (NVIDIA Jetson AGX Orin)
Our co-design strategy shifts the perception pipeline from a **compute-bound** regime to an **IO-Bound (Memory-Bound)** regime.

![Throughput Plateau](assets/throughput_plateau.png)

| Metric | FP16 Baseline | Optimized (INT8 GETA) | Improvement |
| :--- | :--- | :--- | :--- |
| **Throughput** | 60.68 FPS | 141.79 FPS | **+133.6%** |
| **Latency** | 16.48 ms | 7.05 ms | **-57.2%** |
| **Energy/Frame**| 0.594 J | 0.234 J | **-60.6%** |

---

## 🚀 Quick Start
```bash
git clone [https://github.com/Shaima-Alshamiry/hardware-aware-rfdetr-geta.git](https://github.com/Shaima-Alshamiry/hardware-aware-rfdetr-geta.git)
cd hardware-aware-rfdetr
pip install --upgrade pip
pip install -r requirements.txt

## 🎓 Citation
If you use this optimization pipeline or the associated research in your work, please cite both our thesis and the original GETA framework authors:

**Thesis:**
```bibtex
@mastersthesis{alshameri2026hardware,
  title={AI Model Optimization for In-Vehicle Real-Time Applications},
  author={Al-Shameri, Shaima Mohammed Abdulqawi Ghaleb},
  year={2026},
  school={Erasmus Mundus IPCVAI}
}


## GETA Framework

If you find the base optimization framework useful, please cite the original authors:

```bibtex

@article{qu2025automatic,

  title={Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression},

  author={Qu, Xiaoyi and Aponte, David and Banbury, Colby and Robinson, Daniel P and Ding, Tianyu and Koishida, Kazuhito and Zharkov, Ilya and Chen, Tianyi},

  journal={arXiv preprint arXiv:2502.16638},

  year={2025}

}
## 📜 Acknowledgements

This research was conducted as part of the Erasmus Mundus Master in Image Processing and Computer Vision (IPCVAI). Special thanks to Vicomtech for their industrial mentorship and support.

Developed by Shaima Al-Shameri | 2026
