import numpy as np
import matplotlib.pyplot as plt


def main():
    # TODO: Draw 4 graphs in total in one figure
    # TODO: For data to put in graphs, just create your own random data with numpy
    fig, axes = plt.subplots(2, 2)

    # TODO: Scatter Graph
    x = np.random.randn(50)
    y = np.random.randn(50)
    colors = np.random.randint(0, 100, 50)
    sizes = 500 * np.pi * np.random.rand(50) ** 2
    axes[0, 0].scatter(x, y, c=colors, s=sizes, alpha=0.3)

    # TODO: Bar Graph
    x = np.arange(10)
    axes[0, 1].bar(x, x ** 2)

    # TODO: Multi-Bar Graph
    x = np.random.rand(3)
    y = np.random.rand(3)
    z = np.random.rand(3)
    data = [x, y, z]

    x_ax = np.arange(3)
    for i in x_ax:
        axes[1, 0].bar(x_ax, data[i], bottom=np.sum(data[:i], axis=0))
    axes[1, 0].set_xticks(x_ax)
    axes[1, 0].set_xticklabels(['A', 'B', 'C'])

    # TODO: Histogram Graph
    data = np.random.randn(1000)
    axes[1][1].hist(data, bins=40)

    # TODO: Show the image and save it to png file
    plt.show()
    # plt.savefig("sample1.png")

if __name__ == "__main__":
    main()
