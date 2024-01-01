import os
import json
import torch
import argparse
from collections import OrderedDict, defaultdict

from model import SentenceVAE
from utils import *


def main(args):
    datasets = OrderedDict()
    dataset = create_dataset(args=args, split="None")
    params = dict(
        vocab_size=dataset.vocab_size(),
        sos_idx=dataset.sos_idx(),
        eos_idx=dataset.eos_idx(),
        pad_idx=dataset.pad_idx(),
        unk_idx=dataset.unk_idx(),
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
    )
    model = SentenceVAE(**params)

    if not os.path.exists(args.save_model_path):
        raise FileNotFoundError(args.save_model_path)

    model.load_state_dict(torch.load(os.path.join(args.save_model_path, args.model)))
    print("Model loaded from %s" % args.save_model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    samples, z = model.inference(n=args.num_samples)
    print("----------SAMPLES----------")
    print(*idx2word(samples, i2w=i2w, pad_idx=datasets["train"].pad_idx()), sep="\n")

    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    samples, _ = model.inference(z=z)
    print("-------INTERPOLATION-------")
    print(*idx2word(samples, i2w=i2w, pad_idx=datasets["train"].pad_idx()), sep='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_samples", type=int, default=10)

    parser.add_argument("--dataset", type=str, default="minipile")
    parser.add_argument("--s", type=int, default=0)
    parser.add_argument("--max_sequence_length", type=int, default=64)
    parser.add_argument("--test", action="store_true")

    parser.add_argument("-ep", "--epochs", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)

    parser.add_argument("-eb", "--embedding_size", type=int, default=300)
    parser.add_argument("-rnn", "--rnn_type", type=str, default="gru")
    parser.add_argument("-hs", "--hidden_size", type=int, default=256)
    parser.add_argument("-nl", "--num_layers", type=int, default=1)
    parser.add_argument("-bi", "--bidirectional", action="store_true")
    parser.add_argument("-ls", "--latent_size", type=int, default=128)
    parser.add_argument("-wd", "--word_dropout", type=float, default=0)
    parser.add_argument("-ed", "--embedding_dropout", type=float, default=0.5)

    parser.add_argument("-af", "--anneal_function", type=str, default="logistic")
    parser.add_argument("-k", "--k", type=float, default=0.0025)
    parser.add_argument("-x0", "--x0", type=int, default=2500)

    parser.add_argument("-ld", "--log_dir", type=str, default="output")
    parser.add_argument("-bin", "--save_model_path", type=str, default="models")
    parser.add_argument("-m", "--model", type=str, default="3/E0.pytorch")

    arguments = parser.parse_args()
    assert arguments.rnn_type in ["rnn", "lstm", "gru"]
    assert 0 <= arguments.word_dropout <= 1

    main(arguments)
