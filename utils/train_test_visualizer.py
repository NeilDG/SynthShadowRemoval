import yaml
from yaml import SafeLoader
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

def add_plot(losses_dict, key, label, color_index):
    colors = ['r', 'g', 'black', 'darkorange', 'olive', 'palevioletred', 'rosybrown', 'cyan', 'slategray', 'darkmagenta', 'linen', 'chocolate']
    iterations = losses_dict[key]
    loss_values = losses_dict[key]

    x = []
    y = []
    for i in range(0, len(iterations)):
        iteration = list(iterations[i].keys())[0]
        loss_value = list(loss_values[i].values())[0]

        x.append(iteration)
        y.append(loss_value)

    print(x)
    plt.plot(x, y, colors[color_index], label=label)

def main():
    plot_loss_path = "X:/GithubProjects/BMNet/reports/train_test_loss.yaml"
    with open(plot_loss_path) as f:
        losses_dict = yaml.load(f, SafeLoader)

    add_plot(losses_dict, "train", "Train (synth) loss", 0)
    add_plot(losses_dict, "test_istd", "Test (istd) loss", 1)
    plt.legend(loc = 'lower right')

    plt.show()


if __name__ == "__main__":
    main()