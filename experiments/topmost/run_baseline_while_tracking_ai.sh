#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2
###########################################################################
# Run monitoring scripts in the background
# remove file if exists
rm /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/traco/ai/cpu_usage.csv
checkcpu.sh > /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/traco/ai/cpu_usage.csv &
CPU_PID=$!

rm /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/traco/ai/gpu_usage.csv
checkgpu.sh > /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/traco/ai/gpu_usage.csv &
GPU_PID=$!

# Run the Python script and wait for it to complete
source /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/.venv_baseline/bin/activate
python baselines_ai_traco.py

# Kill the monitoring scripts after Python script completes
kill $CPU_PID $GPU_PID

# Ensure processes are terminated
wait $CPU_PID $GPU_PID 2>/dev/null

echo "Baselines execution for TRACO AI completed. CPU and GPU usage saved."

###########################################################################
# Run monitoring scripts in the background
rm /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/hyperminer/ai/cpu_usage.csv
checkcpu.sh > /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/hyperminer/ai/cpu_usage.csv &
CPU_PID=$!

rm /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/hyperminer/ai/gpu_usage.csv
checkgpu.sh > /export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM/data/models/hyperminer/ai/gpu_usage.csv &
GPU_PID=$!

# Run the Python script and wait for it to complete
python baselines_ai_hyperminer.py

# Kill the monitoring scripts after Python script completes
kill $CPU_PID $GPU_PID

# Ensure processes are terminated
wait $CPU_PID $GPU_PID 2>/dev/null

echo "Baselines execution for Hyperminer AI completed. CPU and GPU usage saved."

# HLDA / HDP Baselines
BASE_DIR="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM"
DATA_DIR="$BASE_DIR/data/models"
SCRIPT_DIR="$BASE_DIR/experiments/hlda"
PYTHON_SCRIPT="$SCRIPT_DIR/train_hlda.py"
VENV="$SCRIPT_DIR/.venv_tomotopy/bin/activate"

# Arrays for configuration
CORPORA=("CORDIS" "Cancer" "S2CS-AI")
MODELS=("hlda")

# Function to run a task
run_task() {
    local corpus_name=$1
    local model_type=$2
    local model_dir=$DATA_DIR/$model_type/$(echo "$corpus_name" | tr '[:upper:]' '[:lower:]')

    # Ensure directory exists
    mkdir -p "$model_dir"

    # Remove existing CPU usage file
    local cpu_file="$model_dir/cpu_usage.csv"
    rm -f "$cpu_file"

    # Start CPU monitoring in the background
    checkcpu.sh > "$cpu_file" &
    CPU_PID=$!

    # Activate virtual environment and run the Python script
    source "$VENV"
    python "$PYTHON_SCRIPT" --corpus_name "$corpus_name" --model_type "$model_type"

    # Kill the CPU monitoring process
    kill "$CPU_PID"
    wait "$CPU_PID" 2>/dev/null

    echo "Baselines execution for $model_type $corpus_name completed. CPU and GPU usage saved."
}

# Loop through configurations
for model in "${MODELS[@]}"; do
    for corpus in "${CORPORA[@]}"; do
        run_task "$corpus" "$model"
    done
done


