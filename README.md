# Parallel-Vision-Processor

> This repository contains the draft implementation of parallel vision mamba pipeline network completely developed in C/C++ using the HLS tool. The parallel vision mamba is basically a replacement or we can say in broader sense an enhancement over vision transformer.

> The pipeline uses a unique state space model (SSM) in order to make the model weights depend on the input vector. Mamba layer pipeline architecture is shown below.
<img width="1011" height="620" alt="mamba" src="https://github.com/user-attachments/assets/55382061-35c7-41f6-bfd5-6983e49ffa10" />


> Parallel vision Mamba is nothing but a parallelism implemented with instantiations of four similar mamba blocks.
<img width="699" height="428" alt="parallel vision mamba" src="https://github.com/user-attachments/assets/bd8a2ff1-c41c-4987-b679-4435b97b0db8" />
