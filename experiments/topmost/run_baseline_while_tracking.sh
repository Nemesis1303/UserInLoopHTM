#!/bin/bash

# Run monitoring scripts in the background
checkcpu.sh > /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/hyperminer/cordis/cpu_usage.csv &
CPU_PID=$!

checkgpu.sh > /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/hyperminer/cordis/gpu_usage.csv &
GPU_PID=$!

# Run the Python script and wait for it to complete
python baselines_cordis_hyperminer.py

# Kill the monitoring scripts after Python script completes
kill $CPU_PID $GPU_PID

# Ensure processes are terminated
wait $CPU_PID $GPU_PID 2>/dev/null

echo "Baselines CORDIS execution completed. CPU and GPU usage saved."