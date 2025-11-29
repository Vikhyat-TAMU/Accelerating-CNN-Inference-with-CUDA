# CUDA-Based LeNet-5 Inference Implementation

I led the design and development of this project as part of **ECEN 489: GPU Programming and Visualization**, focusing on accelerating convolutional neural network (CNN) inference using NVIDIA CUDA.  
The work involved building the **LeNet-5** architecture from scratch, implementing both CPU (C) and GPU (CUDA C++) versions, and performing detailed performance benchmarking and memory optimization.  

The goal was to explore how **data parallelism** and **memory hierarchy optimization**especially shared-memory and coalesced global access—affect performance in deep-learning workloads.  
The project demonstrates end-to-end understanding of **GPU kernel design, profiling, and throughput analysis**, aligning with real-world AI hardware acceleration workflows.

---

# Project Overview
This project implements the **LeNet-5 CNN forward pass** in **CUDA C++**, comparing performance between CPU and GPU versions.  
Two GPU implementations are provided:
- **Global Memory Variant** – baseline CUDA kernels using direct global access.  
- **Shared Memory Variant** – optimized version minimizing redundant global reads by caching tiles in shared memory.

---

# Network Architecture

| Layer | Type | Output Size |
|:------|:-----|:------------|
| Input | — | 1 × 28 × 28 |
| Conv1 | 6 filters (5 × 5) | 24 × 24 × 6 |
| Pool1 | 2 × 2 max pool | 12 × 12 × 6 |
| Conv2 | 16 filters (5 × 5) | 8 × 8 × 16 |
| Pool2 | 2 × 2 max pool | 4 × 4 × 16 |
| FC1 | Fully Connected (256 → 120) | 120 |
| FC2 | Fully Connected (120 → 84) | 84 |
| FC3 | Fully Connected (84 → 10) | 10 logits |

---

# Features
- Complete LeNet-5 forward-pass inference pipeline  
- Separate **global** and **shared-memory** CUDA kernels for convolution layers  
- Output verification against CPU baseline  
- Layer-wise performance benchmarking using CUDA events & Nsight  
- Speed-up and efficiency analysis for each kernel  
- Modular structure for easy extension to deeper CNNs

---

# Implementation Details
- **Language:** C, CUDA C++  
- **Hardware:** NVIDIA Tesla V100 / A100  
- **Software Tools:** NVCC, Nsight Systems, VS Code, CUDA 11.x  
- **Dataset:** MNIST (test set for inference validation)

---

# Performance Summary

| Layer | CPU (ms) | CUDA Global (ms) | CUDA Shared (ms) | Speed-up (×) |
|:------|---------:|-----------------:|-----------------:|--------------:|
| Conv1 | 0.325 | 0.019 | 0.014 | 23.2 |
| Pool1 | 0.028 | 0.014 | 0.010 | 2.8 |
| Conv2 | 0.746 | 0.021 | 0.017 | 35.0 |
| Pool2 | 0.009 | 0.005 | 0.004 | 2.3 |
| FC Layers (1–3) | 0.146 | 0.103 | 0.091 | 1.6 |

> The shared-memory implementation achieved up to **35× speed-up** over CPU execution with an average **6.6× overall acceleration**.

---
