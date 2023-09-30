import torch


def preprocess(solution):
    return torch.reshape(torch.clamp(torch.round(torch.from_numpy(solution)).type(torch.int64), min=0, max=50256),
                         (1, -1))


def generate_text(model, tokenizer, prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
