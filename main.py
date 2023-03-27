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

    base_model = LlamaForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=True,
        device_map='auto',
    )

    pipe = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=0.95,
        repetition_penalty=1.2
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    template = """Below is an instruction that describes a task. 
    Write a response that appropriately completes the request.

    ### Instruction: 
    {instruction}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["instruction"])

    llm_chain = LLMChain(prompt=prompt, llm=local_llm)

    question = "What is the capital of England?"

    print(llm_chain.run(question))

