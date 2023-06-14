import torch
import yaml
from tqdm import tqdm
from yaml import SafeLoader
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import global_config
from loaders import dataset_loader


def add_plot(losses_dict, key, label, color_index):
    colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']
    iterations = losses_dict[key]
    loss_values = losses_dict[key]

    x = []
    y = []
    max_steps = 130
    print("Iteration length: ", len(iterations), "Min: ", max_steps)
    for i in range(0, min(max_steps, len(iterations))):
        iteration = list(iterations[i].keys())[0]
        loss_value = list(loss_values[i].values())[0]

        x.append(iteration)
        y.append(loss_value)

    # print(x)
    plt.plot(x, y, colors[color_index], label=label)

def main():
    plot_loss_path = "X:/GithubProjects/BMNet/reports/train_test_loss.yaml"
    with open(plot_loss_path) as f:
        losses_dict = yaml.load(f, SafeLoader)

    add_plot(losses_dict, "train", "Train (synth) - BMNet", 0)
    add_plot(losses_dict, "test_istd", "Test (istd) - BMNet", 1)

    plot_loss_path = "X:/GithubProjects/SG-ShadowNet/reports/train_test_loss.yaml"
    with open(plot_loss_path) as f:
        losses_dict = yaml.load(f, SafeLoader)

    add_plot(losses_dict, "train", "Train (synth) - SG-ShadowNet", 2)
    add_plot(losses_dict, "test_istd", "Test (istd) - SG-ShadowNet", 3)

    plot_loss_path = "X:/GithubProjects/NeuralNets-Experiment3/reports/train_test_loss.yaml"
    with open(plot_loss_path) as f:
        losses_dict = yaml.load(f, SafeLoader)
    add_plot(losses_dict, "train", "Train (synth) - Ours", 4)
    add_plot(losses_dict, "test_istd", "Test (istd) - Ours", 5)

    plt.legend(loc = 'lower right', bbox_to_anchor=(1.04, 0.5))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()