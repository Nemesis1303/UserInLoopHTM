#!/bin/bash
# Combined Script for Training Models: HLDA, HDP, Traco, and HyperMiner

BASE_DIR="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM"
DATA_DIR="$BASE_DIR/data/models_v5_topmost"

# HLDA/HDP Configuration
SCRIPT_DIR="$BASE_DIR/experiments/topmost"
PYTHON_SCRIPT="$SCRIPT_DIR/baselines_topmost.py"
VENV="$BASE_DIR/.venv_baseline/bin/activate"
MODELS=("traco") #"hyperminer"

CORPORA=("CORDIS" "Cancer" "S2CS-AI")

# Function to monitor CPU/GPU usage and execute the task
run_task() {
    local corpus_name=$1
    local model_type=$2
    local python_script=$3
    local venv=$4
    local iteration=$5

    local model_dir=$DATA_DIR/$model_type/$(echo "$corpus_name" | tr '[:upper:]' '[:lower:]')

    # Ensure directory exists
    mkdir -p "$model_dir"

    # Define paths for CPU, GPU usage files, and execution times
    local cpu_file="$model_dir/cpu_usage_iter.$iteration.csv"
    local gpu_file="$model_dir/gpu_usage_iter.$iteration.csv"
    local times_file="$model_dir/execution_times.csv"

    # Initialize the execution times file with a header if it doesn't exist
    if [[ ! -f "$times_file" ]]; then
        echo "Model,Corpus,Iteration,Execution Time (s)" > "$times_file"
    fi

    # Remove existing usage files for the iteration
    rm -f "$cpu_file" "$gpu_file"

    # Start CPU monitoring in the background
    checkcpu.sh > "$cpu_file" &
    CPU_PID=$!

    # Start GPU monitoring in the background
    checkgpu.sh > "$gpu_file" &
    GPU_PID=$!

    # Activate virtual environment and run the Python script
    echo "Activating virtual environment..."
    echo $venv
    source "$venv"

    local start_time=$(date +%s)
    python "$python_script" --corpus_name "$corpus_name" --model_type "$model_type" --iteration "$iteration"
    local end_time=$(date +%s)

    # Calculate execution time
    local execution_time=$((end_time - start_time))

    # Log execution time
    echo "$model_type,$corpus_name,$iteration,$execution_time" >> "$times_file"

    # Kill monitoring processes
    kill "$CPU_PID" "$GPU_PID"
    wait "$CPU_PID" 2>/dev/null
    wait "$GPU_PID" 2>/dev/null

    echo "Execution for $model_type on $corpus_name (Iteration $iteration) completed in $execution_time seconds."
    echo "CPU usage saved to $cpu_file"
    echo "GPU usage saved to $gpu_file"
    echo "Execution times saved to $times_file"
}

# Training Loop
for iteration in {1..3}; do
    for model in "${MODELS[@]}"; do
        for corpus in "${CORPORA[@]}"; do
            echo "Starting $model on $corpus (Iteration $iteration, Traco/HyperMiner)..."
            run_task "$corpus" "$model" "$PYTHON_SCRIPT" "$VENV" "$iteration"

        done
    done
done
