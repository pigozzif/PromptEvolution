import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

from utilities import preprocess, generate_text


def testQA():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # model_name = "t5-base"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # input_ids = tokenizer("Transform the following sentence into its passive form: I eat an apple", return_tensors="pt").input_ids
    # outputs = model.generate(input_ids)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    input_ids = tokenizer.encode("That Italian restaurant is", return_tensors="pt")
    output = model.generate(input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)  # num_return_sequences=1)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    testQA()
    # model_name = "gpt2"
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    # _ = model.eval()
    # print(generate_text(model=model, tokenizer=tokenizer, prompt="tell me what's the capital city of the USA"))
    # inputs = preprocess(np.random.randint(0, 50256, 10))
    # outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
