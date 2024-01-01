import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def read_files(path):
    data = None
    for file in os.listdir(path):
        if not file.endswith("txt") or "minipile" not in file:
            continue
        d = pd.read_csv(os.path.join(path, file), sep=";")
        d["seed"] = int(file.split(".")[1])
        d["elbo.mean"] = d.apply(lambda row: float(row["elbo.mean"].split(",")[0].split("(")[1]), axis=1)
        for col in ["batch.loss", "batch.nll.loss", "batch.kl.loss", "batch.kl.weight", "elbo.mean"]:
            d[col] = d[col].astype(np.float64)
        if data is None:
            data = pd.DataFrame(columns=d.columns)
        data = pd.concat([data, d])
    return data


def plot_vars(data, vs):
    fig, axes = plt.subplots(figsize=(20, 5), nrows=1, ncols=len(vs))
    for col, var in enumerate(vs):
        for split, traj in data.groupby(["split"]):
            median = traj.groupby(traj.epoch)[var].median()
            axes[col].plot(median, label=split)
            err = traj.groupby(traj.epoch)[var].std()
            axes[col].fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
        # axes[row][col].set_title(split, fontsize=15)
        axes[col].set_xlabel("epochs", fontsize=15)
        axes[col].legend()
        axes[col].set_ylabel(var, fontsize=15)


if __name__ == "__main__":
    data = read_files(os.path.join(os.getcwd(), "output"))
    plot_vars(data, ["batch.loss", "batch.nll.loss", "batch.kl.loss", "elbo.mean"])
    plt.savefig("results.png")
    plt.clf()
