# COMPASS: Compressed Processing and Adaptive Scheduling for Sparse Matrix-Matrix Multiplication

This repository contains scripts and utilities for experimenting with a **tile-compressed sparse matrix format** (TC-SpMM) for GPU-accelerated sparse matrix multiplication. The project includes theoretical models, compression utilities, and benchmarking scripts to compare with standard sparse formats (COO, CSR, ELL, BCSR) using NVIDIA cuSPARSE and 64×64 block-based multiplication.  

## Repository Structure

- `compress.py`  
  Computes compression ratios for sparse matrices in the tile-compressed format and compares with standard formats such as COO and CSR. Outputs storage statistics and tile-level information.

- `theoretical_model.py`  
  Implements a theoretical FLOP and memory access model for TC-SpMM and baseline approaches. Calculates expected reduction in computation and memory traffic using tile reuse strategies (row-equal, column-equal, elsewhere-equal).

- `test_{compression_method}`  
  Contains scripts to run sparse matrix multiplication experiments on GPU using different formats:  
  - **BCSR (Block CSR)**  
  - **COO (Coordinate Format)**  
  - **CSR (Compressed Sparse Row)**  
  - **ELL (ELLPACK Format)**  
  - **64×64 CSR Block Method**  
  Scripts collect performance metrics such as execution time, GFLOPs, and memory bandwidth.
