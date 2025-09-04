import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import torch
import json
import os

save_dir = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(save_dir, exist_ok=True)
gpu = torch.cuda.get_device_name()


def plot_bars(title, timings):
    tags = []
    torch_times, fused_times = [], []
    torch_colors, fused_colors = [], []

    for tag in timings.keys():
        timing = timings[tag]
        tags.extend([f"{tag}\nforward", f"{tag}\nbackward"])
        torch_times.extend([timing[0], timing[2]])
        fused_times.extend([timing[1], timing[3]])
        torch_colors.extend(["C0", "C2"])
        fused_colors.extend(["C1", "C3"])

    x = np.arange(len(tags))
    width = 0.4

    matplotlib.rc('font', size=16)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, torch_times, width, color=torch_colors)
    ax.bar(x + width/2, fused_times, width, color=fused_colors)

    for i, (torch_time, fused_time) in enumerate(zip(torch_times, fused_times)):
        ratio = torch_time / fused_time
        ax.annotate(f"{ratio:.1f}x" if ratio < 99.95 else f"{ratio:.3g}x",
                    xy=(i+0.5*width, fused_time+0.01*max(torch_times)),
                    ha='center', va='bottom')

    ax.set_ylabel("Time [ms]")
    # ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(tags)

    ax.bar(np.nan, np.nan, label="PyTorch forward", color="C0")
    ax.bar(np.nan, np.nan, label="Fused forward", color="C1")
    ax.bar(np.nan, np.nan, label="PyTorch backward", color="C2")
    ax.bar(np.nan, np.nan, label="Fused backward", color="C3")
    ax.legend()

    # ax.grid()

    plt.tight_layout()
    # plt.show()

    title = f"{title.replace(' ', '_')}-{gpu.lower().replace(' ', '-')}.png"
    save_path = os.path.join(save_dir, title.replace(' ', '_')+'.png')
    plt.savefig(save_path)


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), "timings.json"), 'r') as fp:
        data = json.load(fp)

    for title, timings in data.items():
        plot_bars(title, timings)
