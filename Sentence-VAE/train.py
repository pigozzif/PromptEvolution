import os
import time
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ..listener import FileListener
from utils import to_var, create_dataset
from model import SentenceVAE


def train_vae(args, listener):
    ts = time.strftime("%Y-%b-%d-%H:%M:%S", time.gmtime())

    splits = ["train", "valid"] + (["test"] if args.test else [])

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = create_dataset(args=args, split=split)
    params = dict(
        vocab_size=datasets["train"].vocab_size(),
        sos_idx=datasets["train"].sos_idx(),
        eos_idx=datasets["train"].eos_idx(),
        pad_idx=datasets["train"].pad_idx(),
        unk_idx=datasets["train"].unk_idx(),
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

    if torch.cuda.is_available():
        model = model.cuda()
        print("Training on GPU")
    else:
        print("Training on CPU")

    print(model)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == "logistic":
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == "linear":
            return min(1, step / x0)

    NLL = torch.nn.NLLLoss(ignore_index=datasets["train"].pad_idx(), reduction="sum")

    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split == "train",
                pin_memory=torch.cuda.is_available()
            )
            tracker = defaultdict(tensor)
            # Enable/Disable Dropout
            if split == "train":
                model.train()
            else:
                model.eval()
            for iteration, batch in enumerate(data_loader):
                batch_size = batch["input"].size(0)
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)
                # Forward pass
                logp, mean, logv, z = model(batch["input"], batch["length"])
                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                                                       batch['length'], mean, logv, args.anneal_function, step, args.k,
                                                       args.x0)
                loss = (NLL_loss + KL_weight * KL_loss) / batch_size
                # backward + optimization
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                # bookkeeping
                tracker["ELBO"] = torch.cat((tracker["ELBO"], loss.data.view(1, -1)), dim=0)
                listener.listen({"epoch": epoch,
                                 "split": split,
                                 "batch.loss": loss.item(),
                                 "batch.nll.loss": NLL_loss.item(),
                                 "batch.kl.loss": KL_loss.item(),
                                 "batch.kl.weight": KL_weight,
                                 "elbo.mean": tracker["ELBO"].mean()})

        # save checkpoint
        checkpoint_path = os.path.join(save_model_path, "E{}.pytorch".format(epoch))
        torch.save(model.state_dict(), checkpoint_path)
        print("Model saved at {}".format(checkpoint_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="minipile")
    parser.add_argument("--s", type=int, default=0)
    parser.add_argument('--create_data', action='store_true')
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

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ["rnn", "lstm", "gru"]
    assert args.anneal_function in ["logistic", "linear"]
    assert 0 <= args.word_dropout <= 1
    os.makedirs(args.log_dir)
    listener = FileListener(file_name=os.path.join(args.log_dir, ".".join([args.dataset, args.s, "txt"])),
                            header=["epoch", "split", "batch.loss", "batch.nll.loss", "batch.kl.loss",
                                    "batch.kl.weight", "elbo.mean"])
    train_vae(args=args, listener=listener)
