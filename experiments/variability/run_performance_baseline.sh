#!/bin/bash
BASE_DIR="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/UserInLoopHTM"
DATA_DIR="$BASE_DIR/data/models_v3"

SCRIPT_DIR="$BASE_DIR/experiments/hlda"
PYTHON_SCRIPT="$SCRIPT_DIR/train_hlda_hdp.py"
VENV="$SCRIPT_DIR/.venv_tomotopy/bin/activate"
MODELS=("hpam")

CORPORA=("CORDIS" "Cancer" "S2CS-AI")

# Set number of topics based on the CORPORA used
declare -A N_TOPICS
declare -A N_TOPICS_SECOND
N_TOPICS["CORDIS"]=6
N_TOPICS["Cancer"]=20
N_TOPICS["S2CS-AI"]=20
N_TOPICS_SECOND=10

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

    # Activate virtual environment
    source "$venv"

    local start_time=$(date +%s)

    # Prepare the Python command
    local cmd="python \"$python_script\" --corpus_name \"$corpus_name\" --output_dir \"$model_dir\" --iteration \"$iteration\" --model_type \"$model_type\" --n_first \"${N_TOPICS[$corpus_name]}\" --n_second \"$N_TOPICS_SECOND\""


    # Print the command before execution
    echo "Executing command: $cmd"

    # Run the command
    eval "$cmd" || { echo "Error ejecutando $python_script"; exit 1; }

    local end_time=$(date +%s)

    # Calculate execution time
    local execution_time=$((end_time - start_time))

    # Log execution time
    echo "$model_type,$corpus_name,$iteration,$execution_time" >> "$times_file"

    # Kill monitoring processes
    kill -0 "$CPU_PID" && kill "$CPU_PID"
    kill -0 "$GPU_PID" && kill "$GPU_PID"
    wait "$CPU_PID" 2>/dev/null
    wait "$GPU_PID" 2>/dev/null

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Execution for $model_type on $corpus_name (Iteration $iteration) completed in $execution_time seconds."
    echo "CPU usage saved to $cpu_file"
    echo "GPU usage saved to $gpu_file"
    echo "Execution times saved to $times_file"
}

# Training Loop
for iteration in {1..3}; do
    for model in "${MODELS[@]}"; do
        for corpus in "${CORPORA[@]}"; do
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting $model on $corpus (Iteration $iteration)..."
            run_task "$corpus" "$model" "$PYTHON_SCRIPT" "$VENV" "$iteration"
        done
    done
done