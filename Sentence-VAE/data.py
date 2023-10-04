import os
import io
import json
import abc

import numpy as np
from collections import defaultdict, OrderedDict, Counter
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer, sent_tokenize
from datasets import load_dataset
from transformers import AutoTokenizer


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class TextDataset(Dataset, abc.ABC):

    def _parse_sentence(self, sentence):
        tokenized_sentence = self.word_tokenizer(sentence,
                                                 padding=True,
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


class PTB(TextDataset):

    def __init__(self, data_dir, split, create_data, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get("max_sequence_length", 50)
        self.min_occ = kwargs.get("min_occ", -1)
        self.raw_data_path = os.path.join(data_dir, "ptb." + split + ".txt")
        self.data_file = "ptb." + split + ".json"
        self.vocab_file = "ptb.vocab.json"
        if create_data:
            print("Creating new %s ptb data." % split.upper())
            self._create_data()
        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new." % (
                split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()
        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)
        return {
            "input": np.asarray(self.data[idx]["input"]),
            "target": np.asarray(self.data[idx]["target"]),
            "length": self.data[idx]["length"]
        }

    def vocab_size(self):
        return len(self.w2i)

    def pad_idx(self):
        return self.w2i["<pad>"]

    def sos_idx(self):
        return self.w2i['<sos>']

    def eos_idx(self):
        return self.w2i['<eos>']

    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), "r") as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab["w2i"], vocab["i2w"]

    def _create_data(self):
        if self.split == "train":
            self._create_vocab()
        else:
            self._load_vocab()
        tokenizer = TweetTokenizer(preserve_case=False)
        data = defaultdict(dict)
        with open(self.raw_data_path, 'r') as file:
            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                inp = ["<sos>"] + words
                inp = inp[:self.max_sequence_length]
                target = words[:self.max_sequence_length - 1]
                target = target + ["<eos>"]
                assert len(inp) == len(target), "%i, %i" % (len(inp), len(target))
                length = len(inp)
                inp.extend(["<pad>"] * (self.max_sequence_length - length))
                target.extend(["<pad>"] * (self.max_sequence_length - length))
                inp = [self.w2i.get(w, self.w2i["<unk>"]) for w in inp]
                target = [self.w2i.get(w, self.w2i["<unk>"]) for w in target]
                idx = len(data)
                data[idx]["input"] = inp
                data[idx]["target"] = target
                data[idx]["length"] = length
        with io.open(os.path.join(self.data_dir, self.data_file), "wb") as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode("utf8", "replace"))
        self._load_data(vocab=False)

    def _create_vocab(self):
        assert self.split == "train", "Vocabulary can only be created for training file."
        tokenizer = TweetTokenizer(preserve_case=False)
        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()
        special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
        for i, st in enumerate(special_tokens):
            i2w[i] = st
            w2i[st] = i
        with open(self.raw_data_path, "r") as file:
            for i, line in enumerate(file):
                words = tokenizer.tokenize(line)
                w2c.update(words)
            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)
        assert len(w2i) == len(i2w)
        print("Vocabulary of {} keys created.".format(len(w2i)))
        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), "wb") as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode("utf8", "replace"))
        self._load_vocab()


class Wikipedia(TextDataset):

    def __init__(self, max_length=64):
        self.word_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.data_stream = load_dataset("wikipedia", "20220301.en", beam_runner="DirectRunner", streaming=True)
        self.max_length = max_length
        self.temp_data = []
        self._idx2word = {v: k for k, v in self.word_tokenizer.vocab.items()}

    def __len__(self):
        return 138151688

    def __getitem__(self, item):
        if not self.temp_data:
            self.temp_data = next(iter(self.data_stream["train"]))["text"]
            self.temp_data = sent_tokenize(self.temp_data)
        return self._parse_sentence(sentence=self.temp_data.pop(0))

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

    def __init__(self, max_length=64):
        self.word_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.data_stream = load_dataset("bookcorpus", streaming=True)
        self.max_length = max_length
        self._idx2word = {v: k for k, v in self.word_tokenizer.vocab.items()}

    def __len__(self):
        return 74004228

    def __getitem__(self, item):
        return self._parse_sentence(sentence=next(iter(self.data_stream["train"]))["text"])

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
