#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH --account=ingenuitylabs
#SBATCH --partition=Northstar
# SBATCH --partition=Sasquatch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=%j_llama_finqa.out
#SBATCH --error=%j_llama_finqa.err
#SBATCH --time=100:00:00

# Set the environment variable to use the first GPU
export CUDA_VISIBLE_DEVICES=1

# Print GPU status using nvidia-smi
nvidia-smi



# exp_qwen_insert_s1_style_with_tag

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_s1_wait/Qwen_1B-MATH-*.json"  --save_dir Qwen_1B_insert_tag --results_dir results_Qwen

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_s1_wait/Qwen_3B-MATH-*.json"  --save_dir Qwen_3B_insert_tag --results_dir results_Qwen

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_s1_wait/Qwen_7B-MATH-*.json"  --save_dir Qwen_7B_insert_tag --results_dir results_Qwen

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_s1_wait/Qwen_32B-MATH-*.json"  --save_dir Qwen_32B_insert_tag --results_dir results_Qwen_big

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_32Bs/Qwen_32B-MATH-*.json"  --save_dir Qwen_32B_argmax --results_dir results_Qwen_big

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_s1_wait/QWQ-MATH-*.json"  --save_dir QWQ_insert_tag --results_dir results_Qwen_big

python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_32Bs/QWQ-MATH-*.json"  --save_dir QWQ_argmax --results_dir results_Qwen_big



# exp_qwen_insert

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_insert_CoT/Qwen_1B-MATH-*.json"  --save_dir Qwen_1B_insert_CoT --results_dir results_Qwen

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_insert_stepbystep/Qwen_1B-MATH-*.json"  --save_dir Qwen_1B_insert_sbs --results_dir results_Qwen

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_insert_termandformula/Qwen_1B-MATH-*.json"  --save_dir Qwen_1B_insert_term --results_dir results_Qwen

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen_insert_endletmmedouble/Qwen_3B-MATH-*.json"  --save_dir Qwen_3B_insert_enddoublecheck --results_dir results_Qwen




# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen/Qwen_1B-MATH-*.json"  --save_dir Qwen_1B --results_dir results_Qwen

# python3 evaluate_general.py --dataset GSM --responses "exp_Qwen/Qwen_1B-GSM-*.json"  --save_dir Qwen_1B --results_dir results_Qwen

# python3 evaluate_general.py --dataset MATH --responses "exp_Qwen/Qwen_1B_base-MATH-*.json"  --save_dir Qwen_1B_base --results_dir results_Qwen

# python3 evaluate_general.py --dataset GSM --responses "exp_Qwen/Qwen_1B_base-GSM-*.json"  --save_dir Qwen_1B_base --results_dir results_Qwen


# Use $1 to accept the task name as an input parameter

# python3 evaluate.py --level 1 --responses "exp_LLAMA_5/LLAMA_1B-l1-*.json"  --save_dir LLAMA_5_1B --results_dir results_LLAMA

# python3 evaluate.py --level 1 --responses "exp_LLAMA_5_8b/LLAMA_8B-l1-*.json"  --save_dir LLAMA_5_8B --results_dir results_LLAMA

# python3 evaluate.py --level 2 --responses "exp_LLAMA_5_8b/LLAMA_8B-l2-*.json"  --save_dir LLAMA_5_8B --results_dir results_LLAMA

# python3 evaluate.py --level 1 --responses "exp_PRM_5_8b/LLAMA_8B-l1-*.json"  --save_dir PRM_5_8B --results_dir results_PRM

# python3 evaluate.py --level 2 --responses "exp_PRM_5_8b/LLAMA_8B-l2-*.json"  --save_dir PRM_5_8B --results_dir results_PRM

# --------------num_branches eval

# python3 evaluate.py --level 2 --responses "exp_PRM_2/LLAMA_1B-l2-*.json"  --save_dir 1b-2 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_4/LLAMA_1B-l2-*.json"  --save_dir 1b-4 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_6/LLAMA_1B-l2-*.json"  --save_dir 1b-6 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_8/LLAMA_1B-l2-*.json"  --save_dir 1b-8 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_12/LLAMA_1B-l2-*.json"  --save_dir 1b-12 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_16/LLAMA_1B-l2-*.json"  --save_dir 1b-16 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_2/LLAMA_3B-l2-*.json"  --save_dir 3b-2 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_4/LLAMA_3B-l2-*.json"  --save_dir 3b-4 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_6/LLAMA_3B-l2-*.json"  --save_dir 3b-6 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_8/LLAMA_3B-l2-*.json"  --save_dir 3b-8 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_12/LLAMA_3B-l2-*.json"  --save_dir 3b-12 --results_dir results_branch

# python3 evaluate.py --level 2 --responses "exp_PRM_16/LLAMA_3B-l2-*.json"  --save_dir 3b-16 --results_dir results_branch



# --------------grid eval

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-3-8/LLAMA_1B-l2-*.json"  --save_dir grid-1b-3-8 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-3-8/LLAMA_3B-l2-*.json"  --save_dir grid-3b-3-8 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-3.5-9/LLAMA_1B-l2-*.json"  --save_dir grid-1b-3.5-9 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-3.5-9/LLAMA_3B-l2-*.json"  --save_dir grid-3b-3.5-9 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-4-10/LLAMA_1B-l2-*.json"  --save_dir grid-1b-4-10 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-4-10/LLAMA_3B-l2-*.json"  --save_dir grid-3b-4-10 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-4.5-11/LLAMA_1B-l2-*.json"  --save_dir grid-1b-4.5-11 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-4.5-11/LLAMA_3B-l2-*.json"  --save_dir grid-3b-4.5-11 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-5-12/LLAMA_1B-l2-*.json"  --save_dir grid-1b-5-12 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-5-12/LLAMA_3B-l2-*.json"  --save_dir grid-3b-5-12 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-5.5-13/LLAMA_1B-l2-*.json"  --save_dir grid-1b-5.5-13 --results_dir results_grid

# python3 evaluate.py --level 2 --responses "exp_PRM_5_1b-5.5-13/LLAMA_3B-l2-*.json"  --save_dir grid-3b-5.5-13 --results_dir results_grid


# --------------old eval
# python3 evaluate.py --level 2 --responses "experiments_argmax/llama1b-l2-*.json"  --save_dir argmax

# python3 evaluate.py --level 1 --responses "exp_PRM_5/llama1b-l1-*.json"  --save_dir PRM_5_1B

# python3 evaluate.py --level 1 --responses "exp_LLAMA_5/llama1b-l1-*.json"  --save_dir LLAMA_5_1B

# python3 evaluate.py --level 2 --responses "exp_random/LLAMA_1B-l2-*.json"  --save_dir LLAMA_random_1b

# python3 evaluate.py --level 2 --responses "exp_random/LLAMA_3B-l2-*.json"  --save_dir LLAMA_random_3b

# python3 evaluate.py --level 2 --responses "exp_LLAMA_5/LLAMA_1B-l2-*.json"  --save_dir LLAMA_5_1B

# python3 evaluate.py --level 1 --responses "exp_LLAMA_5_3b/LLAMA_3B-l1-*.json"  --save_dir LLAMA_5_3B

# python3 evaluate.py --level 2 --responses "exp_LLAMA_5_3b/LLAMA_3B-l2-*.json"  --save_dir LLAMA_5_3B

