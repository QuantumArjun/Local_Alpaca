from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

import argparse

import torch

if __name__ == "__main__":

    #Takes in various model parameters as arguments
    parser = argparse.ArgumentParser(description='Model Variables')
    parser.add_argument('--model', default="chavinlo/alpaca-native")
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.6)
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    #Creates a HuggingFace model from the specified model
    base_model = LlamaForCausalLM.from_pretrained(
        args.model,
        # load_in_8bit=True,
        # device_map='auto',
    )

    #Uses the provided model, tokeinzer, and model parameters in order to make a HF pipeline
    pipe = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=0.95,
        repetition_penalty=1.2
    )

    #Instantiates HF Pipeline
    local_llm = HuggingFacePipeline(pipeline=pipe)

    #Play around - The default instruction that is sent to the model 
    template = """Below is an instruction that describes a task. 
    Write a response that appropriately completes the request.

    ### Instruction: 
    {instruction}

    Answer:"""

    #Turns the prompt into a prompt template 
    prompt = PromptTemplate(template=template, input_variables=["instruction"])
    llm_chain = LLMChain(prompt=prompt, llm=local_llm)

    #Test the prompt template on a basic question
    question = "What is your name?"
    print(llm_chain.run(question))

