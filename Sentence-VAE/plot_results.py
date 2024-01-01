import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def read_files(path):
    data = None
    for file in os.listdir(path):
        if not file.endswith("csv") or "summary" in file:
            continue
        d = pd.read_csv(os.path.join(path, file), sep=",")
        d["approach"] = file.split(".")[0]
        d["problem"] = file.split(".")[1]
        d["seed"] = int(file.split(".")[2])
        d = d[d["iter_idx"] != "iter_idx"]
        d["iter_idx"] = d.apply(lambda row: int(row["iter_idx"]), axis=1)
        for var in ["iter_best_fitness", "iter_eval_time", "iter_model_time"]:
            d[var] = d[var].astype(np.float64)
        d["iter_total_time"] = d["iter_eval_time"] + d["iter_model_time"]
        # d["iter_total_time"] = d["iter_total_time"].cumsum()
        if data is None:
            data = pd.DataFrame(columns=d.columns)
        data = pd.concat([data, d])
    return data


def plot_vars(data, vs):
    fig, axes = plt.subplots(figsize=(20, 20), nrows=len(vs), ncols=len(data["problem"].unique()))
    for row, var in enumerate(vs):
        for col, (problem, traj) in enumerate(data.groupby(["problem"])):
            for approach, inner_traj in traj.groupby(["approach"]):
                median = inner_traj.groupby(inner_traj.iter_idx)[var].median()
                axes[row][col].plot(median, label=approach)
                err = inner_traj.groupby(inner_traj.iter_idx)[var].std()
                axes[row][col].fill_between(np.arange(len(median)), median - err, median + err, alpha=0.25)
            axes[row][col].set_title(problem, fontsize=15)
            axes[row][col].set_xlabel("iters", fontsize=15)
            axes[row][col].legend()
            axes[row][col].set_ylabel(var, fontsize=15)


if __name__ == "__main__":
    data = read_files("results")
    plot_vars(data, ["iter_best_fitness", "iter_eval_time", "iter_model_time", "iter_total_time"])
    plt.savefig("results.png")
    plt.clf()
