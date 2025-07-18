#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The name of your Python script
PYTHON_SCRIPT="optim.py"

# Define the models and datasets to test
MODELS=("qwen-4b")
DATASETS=("aime" "gsm8k" "math500")

# Define test parameters
LIMIT=0            # Set to 0 to process the entire dataset
BATCH_SIZE=8      # Batch size for processing

# The base directory where all results will be stored
# Renamed to reflect that these are full dataset runs
BASE_OUTPUT_DIR="all_test_results_full"

# --- Script Logic ---

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    echo "Please make sure the script is in the same directory or provide the correct path."
    exit 1
fi

# Create the base output directory
mkdir -p "$BASE_OUTPUT_DIR"
echo "All results will be saved in the '$BASE_OUTPUT_DIR' directory."
echo "WARNING: Running on full datasets will take a significant amount of time and disk space."

# Record start time
start_time=$(date +%s)

# Loop through each model
for model in "${MODELS[@]}"; do
  # Loop through each dataset
  for dataset in "${DATASETS[@]}"; do
    
    echo "========================================================================"
    echo "Starting test for MODEL: $model, DATASET: $dataset (Full Dataset)"
    echo "========================================================================"
    
    # Create a specific output directory for this test run
    # Format: all_test_results_full/MODEL_NAME/DATASET_NAME
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$model/$dataset"
    mkdir -p "$OUTPUT_DIR"
    
    # Construct the command
    CMD="python3 $PYTHON_SCRIPT \
        --model \"$model\" \
        --dataset \"$dataset\" \
        --limit $LIMIT \
        --batch-size $BATCH_SIZE \
        --output-dir \"$OUTPUT_DIR\""
        
    
    # Define the log file for this run
    LOG_FILE="$OUTPUT_DIR/run.log"
    
    echo "Output directory: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo "Executing command:"
    echo "$CMD"
    echo "------------------------------------------------------------------------"
    
    # Execute the command, redirecting stdout and stderr to the log file
    # The `tee` command allows you to see the output on the console AND save it to the file.
    eval $CMD 2>&1 | tee "$LOG_FILE"
    
    # Check the exit code of the Python script
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Test failed for MODEL: $model, DATASET: $dataset"
        echo "Check the log file for details: $LOG_FILE"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        # You can choose to exit here or continue with the next test
        # exit 1 
    else
        echo "------------------------------------------------------------------------"
        echo "SUCCESS: Test completed for MODEL: $model, DATASET: $dataset"
    fi
    
    echo "" # Add a newline for better readability
  done
done

# Record end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "========================================================================"
echo "All tests completed!"
echo "Total execution time: $(($duration / 3600))h $((($duration / 60) % 60))m $(($duration % 60))s"
echo "Results and logs are stored in the '$BASE_OUTPUT_DIR' directory."
echo "========================================================================"