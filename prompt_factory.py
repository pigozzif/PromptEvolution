from data.instruction_induction.load_data import load_data


class PromptFactory(object):
    DEMO_INFILL = "I instruct my friend to <mask>\nThe friend read the instruction and wrote an output for every one " \
                  "of the inputs. Here are the input-output pairs: "
    DEMO_FORWARD = "I gave a friend an instruction and five inputs. The friend read the instruction and wrote an " \
                   "output for every one of the inputs. Here are the input-output pairs:\n[EXAMPLES]\nThe instruction" \
                   " was "

    def __init__(self, tokenizer, model, sub_task):
        self.tokenizer = tokenizer
        self.model = model
        self.test_data = load_data("eval", sub_task)
        self._fill_demo()

    def _fill_demo(self):
        self.DEMO_INFILL = "\n".join([self.DEMO_INFILL] + ["Input {0} Output {1}".format(i, o[0])
                                                           for i, o in zip(self.test_data[0], self.test_data[1])])
        self.DEMO_FORWARD = self.DEMO_FORWARD.replace("[EXAMPLES]", "\n".join(["Input {0} Output {1}".format(i, o[0])
                                                                               for i, o in zip(self.test_data[0],
                                                                                               self.test_data[
                                                                                                   1])]))

    def create(self):
        inputs = self.tokenizer(self.DEMO_FORWARD, truncation=True, max_length=1024, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=20 + len(inputs.input_ids))
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0][len(self.DEMO_FORWARD):]

    def create_population(self, pop_size):
        inputs = self.tokenizer(self.DEMO_FORWARD, truncation=True, max_length=1024, return_tensors="pt")
        outputs = self.model.generate(**inputs,
                                      num_beams=pop_size,
                                      max_new_tokens=20 + len(inputs.input_ids),
                                      num_return_sequences=pop_size)
        return [text[len(self.DEMO_FORWARD):]
                for text in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
