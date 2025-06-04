from typing import Literal
import tyro
import logging
from openai import OpenAI
from entropix.model import GenerationData
import pandas as pd
import os
import glob, re
from datasets import load_dataset

MY_API_KEYS = ""

# if os.getenv("OPENROUTER_API_KEY") is None:
#     # read .env file into runtime environment variables
#     with open(".env", "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             var = line.strip().split("=", 1)
#             if len(var) == 2: os.environ[var[0]] = var[1]

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=MY_API_KEYS)

def main(dataset: Literal["MATH","GSM"], save_dir: str, results_dir: str, responses: str, out: str | None = None):
    """Evaluate the AI answers to CFA questions.

    Args:
        level (int): The level of the CFA question.
        responses (str): Glob pattern for the GenerationData files to evaluate.
    """
    if dataset == "MATH":
        ds = load_dataset("HuggingFaceH4/MATH-500")
        ds = ds['test']
        ds = ds.rename_column("problem", "question")
    elif dataset == "GSM":
        ds = load_dataset("openai/gsm8k", "main")
        ds = ds['test']
    else:
        ValueError("Wrong dataset name, should pick from MATH, GSM")

    response_files = glob.glob(responses)
    def numeric_sort_key(f):
        match = re.search(r'-(\d+)\.json$', f)
        return int(match.group(1)) if match else float('inf')

    response_files = sorted(response_files, key=numeric_sort_key)
    df = pd.DataFrame(columns=["question", "answer", "ai_answer", "correct", "gen_data"])
    for (i, row), file in zip(enumerate(ds), response_files):
        while True:  # Keep retrying until successful
            try:
                gen = GenerationData.load(file)
                context = f"""Question:
                            {row['question']} /n/n

                            Correct Choice:
                            {row['answer']} /n/n

                            Student Choice:
                            {gen.response}"""
                
                print(context, flush= True)
                completion = client.chat.completions.create(
                    # https://openrouter.ai/models
                    model="meta-llama/llama-3.3-70b-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are evaluating student answers. You will be provided with the question, the correct Choice, and the student's Choice. 
                            Determine whether the student's Choice matches the correct Choice. Respond only with "Correct" or "Incorrect" .""",
                        },
                        {"role": "user", "content": context},
                    ],
                )
                eval = completion.choices[0].message.content
                is_correct = None
                if eval and "incorrect" in eval.lower():
                    is_correct = False
                elif eval and "correct" in eval.lower():
                    is_correct = True
                else:
                    logging.error(f"Could not parse evaluation for {dataset} question {i}")
                    print(f"Evaluator response: {eval}")

                new_row = pd.DataFrame([{
                    "question": f"{row['question']}",
                    "answer": f"{row['answer']}",
                    "ai_answer": gen.response,
                    "correct": is_correct,
                    "gen_data": str(gen.to_dict()),
                }])
                df = pd.concat([df, new_row], ignore_index=True)

                print(f"finish evaluating for {dataset} question {i}", flush=True)

                break 

            except Exception as e:
                    print(f"Error encountered for index {i}: {e}. Retrying...", flush=True)

    if out is None:
        os.makedirs(f"{results_dir}", exist_ok=True)
        out = f"{results_dir}/{dataset}_{save_dir}_choice.json"
    df.to_json(out, orient="records")
        
if __name__ == "__main__":
    tyro.cli(main)
