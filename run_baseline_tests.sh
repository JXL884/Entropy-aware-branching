#!/bin/bash
# Comprehensive Baseline Testing Script
# This script runs baseline tests for all model-dataset combinations

set -e  # Exit on any error

# Configuration
MODELS=("qwen-4b" "llama-8b" "deepseek-8b")
DATASETS=("aime" "gsm8k" "math500")
QUESTIONS_PER_TEST=10  # Adjust based on your resources

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_error "Virtual environment not activated. Please run: source .venv/bin/activate"
    exit 1
fi

# Check if required files exist
if [[ ! -f "baseline_test.py" ]]; then
    print_error "baseline_test.py not found in current directory"
    exit 1
fi

# Create results directory
RESULTS_DIR="baseline_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
print_status "Results will be saved to: $RESULTS_DIR"

# Function to run a single test
run_test() {
    local model=$1
    local dataset=$2
    local output_dir="$RESULTS_DIR/results_${model}_${dataset}"
    
    print_status "Testing $model on $dataset dataset..."
    
    # Run the test
    python baseline_test.py \
        --model "$model" \
        --dataset "$dataset" \
        --limit "$QUESTIONS_PER_TEST" \
        --output-dir "$output_dir" \
        --save-plots
    
    if [[ $? -eq 0 ]]; then
        print_success "Completed: $model on $dataset"
    else
        print_error "Failed: $model on $dataset"
        return 1
    fi
}

# Function to run all tests
run_all_tests() {
    local total_tests=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
    local current_test=0
    
    print_status "Starting comprehensive baseline testing..."
    print_status "Total tests to run: $total_tests"
    print_status "Questions per test: $QUESTIONS_PER_TEST"
    
    # Create a log file
    LOG_FILE="$RESULTS_DIR/test_log.txt"
    echo "Baseline Testing Log - $(date)" > "$LOG_FILE"
    echo "Total tests: $total_tests" >> "$LOG_FILE"
    echo "Questions per test: $QUESTIONS_PER_TEST" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    for model in "${MODELS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            current_test=$((current_test + 1))
            print_status "Progress: $current_test/$total_tests - $model on $dataset"
            
            # Log the test start
            echo "[$(date)] Starting: $model on $dataset" >> "$LOG_FILE"
            
            # Run the test
            if run_test "$model" "$dataset"; then
                echo "[$(date)] Completed: $model on $dataset" >> "$LOG_FILE"
            else
                echo "[$(date)] Failed: $model on $dataset" >> "$LOG_FILE"
                print_warning "Test failed, continuing with next test..."
            fi
            
            echo "" >> "$LOG_FILE"
        done
    done
    
    print_success "All tests completed!"
}

# Function to analyze results
analyze_results() {
    print_status "Analyzing results..."
    
    # Find all result directories
    local result_dirs=()
    for dir in "$RESULTS_DIR"/results_*; do
        if [[ -d "$dir" ]]; then
            result_dirs+=("$dir")
        fi
    done
    
    if [[ ${#result_dirs[@]} -eq 0 ]]; then
        print_warning "No result directories found for analysis"
        return
    fi
    
    # Run analysis
    python analyze_results.py \
        "${result_dirs[@]}" \
        --output-dir "$RESULTS_DIR/analysis" \
        --save-csv
    
    if [[ $? -eq 0 ]]; then
        print_success "Analysis completed! Check $RESULTS_DIR/analysis/"
    else
        print_error "Analysis failed"
    fi
}

# Function to create summary report
create_summary() {
    print_status "Creating summary report..."
    
    SUMMARY_FILE="$RESULTS_DIR/summary_report.txt"
    
    {
        echo "BASELINE TESTING SUMMARY REPORT"
        echo "================================"
        echo "Date: $(date)"
        echo "Total tests run: $(( ${#MODELS[@]} * ${#DATASETS[@]} ))"
        echo "Questions per test: $QUESTIONS_PER_TEST"
        echo ""
        echo "Models tested:"
        for model in "${MODELS[@]}"; do
            echo "  - $model"
        done
        echo ""
        echo "Datasets tested:"
        for dataset in "${DATASETS[@]}"; do
            echo "  - $dataset"
        done
        echo ""
        echo "Results directory: $RESULTS_DIR"
        echo ""
        echo "To analyze results, run:"
        echo "python analyze_results.py $RESULTS_DIR/results_* --output-dir $RESULTS_DIR/analysis"
        echo ""
        echo "Individual test results:"
        for dir in "$RESULTS_DIR"/results_*; do
            if [[ -d "$dir" ]]; then
                echo "  - $dir"
            fi
        done
    } > "$SUMMARY_FILE"
    
    print_success "Summary report created: $SUMMARY_FILE"
}

# Main execution
main() {
    print_status "Starting baseline testing framework..."
    
    # Check system resources
    print_status "Checking system resources..."
    
    # Check available GPU memory (if using GPU)
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        print_status "Available GPU memory: ${GPU_MEM}MB"
        
        if [[ $GPU_MEM -lt 8000 ]]; then
            print_warning "Low GPU memory detected. Consider reducing QUESTIONS_PER_TEST"
        fi
    fi
    
    # Run all tests
    run_all_tests
    
    # Analyze results
    analyze_results
    
    # Create summary
    create_summary
    
    print_success "Baseline testing completed successfully!"
    print_status "Results saved in: $RESULTS_DIR"
    print_status "Check the summary report: $RESULTS_DIR/summary_report.txt"
}

# Handle script arguments
case "${1:-}" in
    "test")
        # Run a single test
        if [[ $# -lt 3 ]]; then
            print_error "Usage: $0 test <model> <dataset>"
            exit 1
        fi
        run_test "$2" "$3"
        ;;
    "analyze")
        # Only analyze existing results
        analyze_results
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  Run all tests"
        echo "  test       Run a single test: $0 test <model> <dataset>"
        echo "  analyze    Analyze existing results"
        echo "  help       Show this help"
        echo ""
        echo "Examples:"
        echo "  $0                                    # Run all tests"
        echo "  $0 test qwen-4b aime                 # Test Qwen-4B on AIME"
        echo "  $0 analyze                           # Analyze existing results"
        ;;
    "")
        # Run all tests (default)
        main
        ;;
    *)
        print_error "Unknown command: $1"
        print_error "Use '$0 help' for usage information"
        exit 1
        ;;
esac 