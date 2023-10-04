import abc

import numpy as np
import nltk
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer


class TextDataset(Dataset, abc.ABC):

    def _parse_sentence(self, sentence):
        tokenized_sentence = self.word_tokenizer(sentence,
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=64,
                                                 add_special_tokens=True).input_ids
        return {"input": np.asarray(tokenized_sentence[:-1]),
                "target": np.asarray(tokenized_sentence[1:]),
                "length": len(tokenized_sentence) - 1}

    @abc.abstractmethod
    def vocab_size(self):
        pass

    @abc.abstractmethod
    def pad_idx(self):
        pass

    @abc.abstractmethod
    def sos_idx(self):
        pass

    @abc.abstractmethod
    def eos_idx(self):
        pass

    @abc.abstractmethod
    def unk_idx(self):
        pass

    @abc.abstractmethod
    def get_w2i(self):
        pass

    @abc.abstractmethod
    def get_i2w(self):
        pass


class Wikipedia(TextDataset):

    def __init__(self, train, val_split=0.1, max_length=64):
        nltk.downlaod("punkt")
        self.word_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.train = train
        self.split = val_split if not train else 1 - val_split
        self.data_stream = load_dataset("wikipedia", "20220301.en", beam_runner="DirectRunner", streaming=True)
        self.max_length = max_length
        self.idx = set(np.random.choice(np.arange(self.__len()), size=len(self), replace=False))
        self.c = 0
        self.temp_data = []
        self._idx2word = {v: k for k, v in self.word_tokenizer.vocab.items()}

    def __len(self):
        return 138151688

    def __len__(self):
        return int(self.__len() * self.split)

    def __getitem__(self, item):
        while True:
            if not self.temp_data:
                self.temp_data = next(iter(self.data_stream["train"]))["text"]
                self.temp_data = nltk.tokenize.sent_tokenize(self.temp_data)
            next_sentence = self.temp_data.pop(0)
            if self.c in self.idx:
                self.c += 1
                break
            self.c += 1
        return self._parse_sentence(sentence=next_sentence)

    def vocab_size(self):
        return self.word_tokenizer.vocab_size

    def pad_idx(self):
        return self.word_tokenizer.pad_token_id

    def sos_idx(self):
        return self.word_tokenizer.cls_token_id

    def eos_idx(self):
        return self.word_tokenizer.sep_token_id

    def unk_idx(self):
        return self.word_tokenizer.unk_token_id

    def get_w2i(self):
        return self.word_tokenizer.vocab

    def get_i2w(self):
        return self._idx2word


class BookCorpus(TextDataset):

    def __init__(self, train, val_split=0.2, max_length=64):
        nltk.downlaod("punkt")
        self.word_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.split = val_split if not train else 1 - val_split
        self.data = load_dataset("bookcorpus")["train"]
        self.max_length = max_length
        self.idx = np.arange(self.__len())
        np.random.shuffle(self.idx)
        self.c = 0 if train else int(self.__len() * (1 - val_split))
        self._idx2word = {v: k for k, v in self.word_tokenizer.vocab.items()}

    def __len(self):
        return 74004228

    def __len__(self):
        return int(self.__len() * self.split)

    def __getitem__(self, item):
        return self._parse_sentence(sentence=self.data[int(self.idx[item + self.c])]["text"])

    def vocab_size(self):
        return self.word_tokenizer.vocab_size

    def pad_idx(self):
        return self.word_tokenizer.pad_token_id

    def sos_idx(self):
        return self.word_tokenizer.cls_token_id

    def eos_idx(self):
        return self.word_tokenizer.sep_token_id

    def unk_idx(self):
        return self.word_tokenizer.unk_token_id

    def get_w2i(self):
        return self.word_tokenizer.vocab

    def get_i2w(self):
        return self._idx2word


class MiniPile(TextDataset):

    def __init__(self, split, max_length=64):
        nltk.downlaod("punkt")
        self.word_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if split == "valid":
            split += "ation"
        self.split = split
        self.data = load_dataset("JeanKaddour/MiniPile")[split]
        self.max_length = max_length
        self.curr_sentences = []
        self.idx = 0
        self._idx2word = {v: k for k, v in self.word_tokenizer.vocab.items()}

    def __len__(self):
        if self.split == "train":
            return 42339515
        elif self.split == "validation":
            return 20203
        return 442833

    def __getitem__(self, item):
        if not self.curr_sentences:
            self.curr_sentences = nltk.tokenize.sent_tokenize(self.data[self.idx]["text"])
            self.idx += 1
        return self._parse_sentence(sentence=self.curr_sentences.pop(0))

    def vocab_size(self):
        return self.word_tokenizer.vocab_size

    def pad_idx(self):
        return self.word_tokenizer.pad_token_id

    def sos_idx(self):
        return self.word_tokenizer.cls_token_id

    def eos_idx(self):
        return self.word_tokenizer.sep_token_id

    def unk_idx(self):
        return self.word_tokenizer.unk_token_id

    def get_w2i(self):
        return self.word_tokenizer.vocab

    def get_i2w(self):
        return self._idx2word
