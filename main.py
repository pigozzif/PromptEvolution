import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.evaluate_instruction_induction import InstructionInductionEvaluator, sub_tasks
from evaluation.metrics import normalize_prediction
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


def test_zero_shot(task, n_prompts=10):
    listener = FileListener(
        file_name=os.path.join("output", ".".join(["zeroshot", sub_task, "txt"])),
        header=["scores", "prompts"])
    evaluator = InstructionInductionEvaluator(model_name="tiiuae/falcon-7b", sub_task=task)
    factory = PromptFactory(tokenizer=evaluator.tokenizer, model=evaluator.model, sub_task=task)
    annotations = [normalize_prediction(pred, lowercase=True)
                   for pred in json.load(open(os.path.join(os.getcwd(),
                                                           "data",
                                                           "instruction_induction",
                                                           "annotations",
                                                           "{}.json".format(
                                                               task))))[
                       "annotations"]]
    prompts = [normalize_prediction(prompt, lowercase=True) for prompt in factory.create_population(
        pop_size=n_prompts)] * len(annotations)
    new_annotations = []
    for annotation in annotations:
        new_annotations.extend([annotation] * n_prompts)
    scores = evaluator.scores_against_gold(prompts=prompts, annotations=new_annotations)
    listener.listen(**{"scores": "/".join([str(s.item()) for s in scores]),
                       "prompts": "/".join([p.replace("/", "*") for p in prompts])})


if __name__ == "__main__":
    # testQA()
    for sub_task in sub_tasks:
        test_zero_shot(task=sub_task)
