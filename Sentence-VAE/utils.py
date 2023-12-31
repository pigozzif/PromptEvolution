import random
import torch
import numpy as np

from data import Wikipedia, BookCorpus, MiniPile, Dummy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def idx2word(idx, i2w, pad_idx):
    sent_str = [str()] * len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[word_id.item()] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str


def interpolate(start, end, steps):
    interpolation = np.zeros((start.shape[0], steps + 2))
    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps + 2)
    return interpolation.T


def create_dataset(args, split):
    if args.dataset == "dummy":
        return Dummy()
    elif args.dataset == "wikipedia":
        return Wikipedia(train=split == "train", max_length=args.max_sequence_length)
    elif args.dataset == "bc":
        return BookCorpus(train=split == "train", max_length=args.max_sequence_length)
    elif args.dataset == "minipile":
        return MiniPile(split=split, max_length=args.max_sequence_length)
    raise ValueError("Invalid dataset: {}".format(args.dataset))
