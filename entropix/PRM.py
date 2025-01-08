from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
import numpy as np
import torch
import argparse

def select_sample(args, sample, model, tokenizer, candidate_tokens):
    prompt = sample['prompt']
    scores_list = []
    answers = sample['branches'][:args.num_n]
    step_scores = []
    
    print(f"Prompt: {prompt}")  # Print the prompt
    
    for idx, ans in enumerate(answers):
        single_step_score = []
        conversation = []
        forward_conv = []
        
        if args.model_type == "Mistral":
            processed_ans = ans.replace(" ки", "")
            conversation.append({"content": prompt + " " + processed_ans, "role": "user"})
        else:
            processed_ans = ans
            conversation.append({"content": prompt + " " + ans, "role": "user"})
        
        conversation.append({"content": "+", "role": "assistant"})
        
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            logits = model(input_ids).logits[:,-3,candidate_tokens]  # Extract logits for the last '-3' position
            scores = logits.softmax(dim=-1)[:,0]  # Get the probability of '+' (index 0 in candidate_tokens)
        
        # Save the scores
        scores_list.append(scores[0].detach().to('cpu', dtype=torch.float32))
        
        # Print the assistant's response and the corresponding probability
        print(f"Response {idx + 1}: {processed_ans}")
        print(f"Probability of '+': {scores[0].item()}")
    
    # Determine the selected response
    idx = scores_list.index(max(scores_list))
    sample['step_scores'] = [x.item() for x in scores_list]  # Add the step_score attribute to each sample
    
    # Print the selected response
    print(f"Selected Response: {answers[idx]}")
    print(f"Step Scores: {sample['step_scores']}\n")
    
    return sample


def process_single_prompt(args, score_model):
    prompt = args.prompt
    responses = args.responses
    # Create sample for PRM
    sample = {
        "prompt": prompt,
        "branches": responses
    }
    
    model = score_model.params
    tokenizer = score_model.tokenizer

    plus_tag_id = tokenizer.encode('+')[-1]
    minus_tag_id = tokenizer.encode('-')[-1]
    candidate_tokens = [plus_tag_id, minus_tag_id]
    
    new_sample = select_sample(args, sample, model, tokenizer, candidate_tokens)
    
    print(f"Prompt: {prompt}")
    print(f"Selected Answer: {sample['branches'][new_sample['step_scores'].index(max(new_sample['step_scores']))]}")
    print(f"Step Scores: {new_sample['step_scores']}")

    return new_sample

# clean this function later
def process_response(prompt, responses, score_model):
    args = argparse.Namespace(
        num_n=1024,
        model_type="Deepseek",
        prompt=prompt,
        responses=responses,
    )

    # Process a single prompt if provided
    new_sample = process_single_prompt(args, score_model)

    return new_sample
