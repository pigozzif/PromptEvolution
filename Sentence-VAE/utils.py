import torch
import numpy as np

from data import PTB, Wikipedia, BookCorpus, MiniPile


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
            sent_str[i] += i2w[str(word_id.item())] + " "
        sent_str[i] = sent_str[i].strip()
    return sent_str


def interpolate(start, end, steps):
    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps + 2)

    return interpolation.T


def create_dataset(args, split):
    if args.dataset == "ptb":
        return PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )
    elif args.dataset == "wikipedia":
        return Wikipedia(train=split == "train", max_length=args.max_sequence_length)
    elif args.dataset == "bc":
        return BookCorpus(train=split == "train", max_length=args.max_sequence_length)
    elif args.dataset == "minipile":
        return MiniPile(split=split, max_length=args.max_sequence_length)
    raise ValueError("Invalid dataset: {}".format(args.dataset))
