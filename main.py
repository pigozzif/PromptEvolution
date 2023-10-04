from transformers import AutoModelForCausalLM, AutoTokenizer


def testQA():
    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    _ = model.eval()

    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(**inputs)  # , num_beams=5,
    # max_new_tokens=32, early_stopping=True,
    # no_repeat_ngram_size=2)
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(text_output[0])


if __name__ == "__main__":
    # testQA()
    print(AutoTokenizer.from_pretrained("bert-base-uncased")("what are you doing? and now",
                                                             padding="max_length",
                                                             truncation=True,
                                                             max_length=64,
                                                             add_special_tokens=True).input_ids)
