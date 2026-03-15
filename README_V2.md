# 🧠 Near-Field Beam Training for XL-MIMO using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-34%2F34%20passing-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A deep learning framework for **near-field beam training** in extremely large-scale MIMO (XL-MIMO) systems, enabling efficient ISAC (Integrated Sensing and Communications) beam management.

**Paper**: J. Nie, **Y. Cui**, et al., "Near-Field Beam Training for Extremely Large-Scale MIMO Based on Deep Learning," *IEEE Trans. Mobile Computing*, 2025.

---

## 🎯 Overview

Traditional beam training in XL-MIMO systems requires exhaustive search over codebooks, which becomes prohibitively expensive in near-field scenarios with both angle AND distance parameters. This repository implements a **CNN-based approach** that learns to predict optimal beams directly from received signals, reducing training overhead by >10x.

## 📊 Results

![Training Convergence](results/p0c_training.png)
![Beam Pattern Comparison](results/p0c_beam_pattern.png)

## 🚀 Quick Start

```bash
# Install
pip install torch numpy scipy matplotlib pytest

# Run tests (34/34 passing)
PYTHONPATH=src pytest tests/ -v

# Generate figures
python generate_figures.py
```

## 📖 Mathematical Background

### Near-Field Channel Model
The near-field spherical wave channel:

h(r,θ) = (α/r) · exp(-j2πr/λ) · a(θ,r)

where r is distance, θ is angle, α is path loss, and a(θ,r) is the near-field steering vector.

### CNN Beam Predictor
The model learns f: Received Signal → Beam Index + Distance, trained with a **rate-driven loss**:

L = -log₂(1 + SNR·|hᴴw|²)

## 🏗️ Project Structure

```
├── src/
│   ├── model.py          # CNN architecture (UNet-like)
│   ├── channel.py        # Near-field channel model
│   ├── beamforming.py    # DFT & polar codebooks
│   ├── trainer.py        # Training pipeline
│   ├── evaluator.py      # Metrics & evaluation
│   └── utils.py          # Trans_vrf, rate_func
├── tests/                # 34 unit tests
├── results/              # Simulation figures
└── legacy/               # Original code (preserved)
```

## 📚 Citation

```bibtex
@article{nie2025near,
  title={Near-Field Beam Training for XL-MIMO Based on Deep Learning},
  author={Nie, Jinghao and Cui, Yuanhao and ...},
  journal={IEEE Trans. Mobile Computing},
  year={2025}
}
```
