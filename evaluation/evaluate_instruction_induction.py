from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from data.instruction_induction.load_data import load_data
from .metrics import *

sub_tasks = ["antonyms", "cause_and_effect", "common_concept", "diff", "first_word_letter",
             "informal_to_formal", "larger_animal", "letters_list", "taxonomy_animal", "negation", "num_to_verbal",
             "active_to_passive", "singular_to_plural", "rhymes",
             "second_word_letter", "sentence_similarity", "sentiment", "orthography_starts_with",
             "sum", "synonyms", "translation_en-de", "translation_en-es",
             "translation_en-fr", "word_in_context"]


class InstructionInductionEvaluator(object):

    PROMPT_TOKEN = "[PROMPT]"
    INPUT_TOKEN = "[INPUT]"
    OUTPUT_TOKEN = "[OUTPUT]"

    def __init__(self, model_name, sub_task):
        assert sub_task in sub_tasks, "Task not found!"
        self.task = sub_task
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Running on GPU")
        else:
            print("Running on CPU")
        _ = self.model.eval()
        self.test_data = load_data("eval", sub_task)
        self._set_score_fn(sub_task)
        self.eval_template = "Instruction: {0}\n\nInput: {1}\nOutput: {2}".format(self.PROMPT_TOKEN,
                                                                                  self.INPUT_TOKEN,
                                                                                  self.OUTPUT_TOKEN)

    def _set_score_fn(self, sub_task, bert=True):
        if bert:
            self.score_fn = get_bert_score
            return
        metric = TASK_TO_METRIC.get(sub_task, default_metric)
        if metric == "f1":
            self.score_fn = get_multi_answer_f1
        elif metric == "es":
            self.score_fn = get_multi_answer_exact_set
        elif metric == "contains":
            self.score_fn = get_multi_answer_contains
        elif metric == "em":
            self.score_fn = get_multi_answer_em

    def _fill_template(self, prompt, i):
        return self.eval_template.replace(self.PROMPT_TOKEN, prompt).replace(self.INPUT_TOKEN, i).\
            replace(self.OUTPUT_TOKEN, "")

    def evaluate_prompt(self, prompt):
        scores = []
        for i, g in zip(self.test_data[0], self.test_data[1]):
            query = self.tokenizer(self._fill_template(prompt, i), return_tensors="pt")
            output = self.tokenizer.decode(self.model.generate(**query)[0], skip_special_tokens=True)
            score = self.score_fn([output], [self.eval_template.replace(self.OUTPUT_TOKEN, g[0])])[0].item()
            scores.append(score)
        return np.mean(scores)

    def evaluate_prompts(self, prompts):
        scores = np.zeros((len(self.test_data[0]), len(prompts)))
        for idx, (i, g) in enumerate(zip(self.test_data[0], self.test_data[1])):
            queries = self.tokenizer([normalize_prediction(self._fill_template(prompt, i), lowercase=True)
                                      for prompt in prompts], return_tensors="pt")
            outputs = self.tokenizer.batch_decode(self.model.generate(**queries), skip_special_tokens=True)
            scores[idx] = self.score_fn(outputs, [self.eval_template.replace(self.OUTPUT_TOKEN, g[0])] * len(outputs))\
                .numpy()
        return np.mean(scores, axis=1)

    def scores_against_gold(self, prompts, annotations):
        return self.score_fn(prompts, annotations)
