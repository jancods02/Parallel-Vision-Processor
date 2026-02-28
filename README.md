# Parallel-Vision-Processor

> This repository contains the draft implementation of parallel vision mamba pipeline network completely developed in C/C++ using the HLS tool. The parallel vision mamba is basically a replacement or we can say in broader sense an enhancement over vision transformer.

> The pipeline uses a unique state space model (SSM) in order to make the model weights depend on the input vector. Mamba layer pipeline architecture is shown below.
![mamba](images/mamba.png)

> Parallel vision Mamba is nothing but a parallelism implemented with instantiations of four similar mamba blocks.
![PVM](images/parallel vision mamba.png)
