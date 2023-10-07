import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.evaluate_instruction_induction import InstructionInductionEvaluator
from listener import FileListener
from prompt_factory import PromptFactory


def testQA():
    model_name = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    _ = model.eval()

    prompt = "Q: What is the largest animal?\nA:"
    inputs = tokenizer(prompt, truncation=True, max_length=1024, return_tensors="pt")

    outputs = model.generate(**inputs)  # , num_beams=5,
    # max_new_tokens=32, early_stopping=True,
    # no_repeat_ngram_size=2)
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(text_output[0])


def test_zero_shot(task, listener, n_prompts=30):
    evaluator = InstructionInductionEvaluator(model_name="tiiuae/falcon-7b", sub_task=task)
    factory = PromptFactory(tokenizer=evaluator.tokenizer, model=evaluator.model, sub_task=task)
    prompts = factory.create_population(pop_size=n_prompts)
    scores = evaluator.scores_against_gold(prompts=prompts)
    listener.listen(**{"scores": "/".join([str(s) for s in scores]),
                       "prompts": "/".join([p.replace("/", "*") for p in prompts])})


if __name__ == "__main__":
    # testQA()
    sub_task = sys.argv[1]
    lis = FileListener(
        file_name=os.path.join("output", ".".join(["zeroshot", sub_task, "txt"])),
        header=["scores", "prompts"])
    test_zero_shot(task=sub_task, listener=lis)
