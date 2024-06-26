import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as transformers_logging
import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def chat(model_id): 

    # Set seed and verbosity
    torch.random.manual_seed(0)
    transformers_logging.set_verbosity(40)

    # Initial model setup
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


    messages = []
    while True:
        user_input = input("You: ")
        messages.append({
            "role": "user",
            "content": user_input
        })

        output = pipe(messages, **generation_args)
        response = output[0]['generated_text']
        print(f"Bot: {response}\n")
        messages.append({
            "role": "assistant",
            "content": response
        })
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide the model ID for the chat script.")
    parser.add_argument("model_id", type=str, help="The ID of the model to load.")
    args = parser.parse_args()

    chat(args.model_id)