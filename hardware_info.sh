#!/bin/bash

echo "CPU Information:"
lscpu | grep "Model name\|CPU(s)\|Thread(s)"

echo -e "\nRAM Information:"
free -h

echo -e "\nGPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=gpu_name --format=csv,noheader
else
    echo "No NVIDIA GPUs detected or nvidia-smi not installed."
fi
