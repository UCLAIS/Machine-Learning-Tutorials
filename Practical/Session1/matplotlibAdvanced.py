import numpy as np
import matplotlib.pyplot as plt


def main():
    # TODO: Draw 4 different graphs in one figure separately - hint: change arguments in plt.subplots
    # TODO: For data to put in graphs, just create your own random data with numpy
    fig, axes = None

    # TODO: Scatter Graph
    x = np.random.randn(50)
    y = np.random.randn(50)


    # TODO: Bar Graph
    x = np.arange(10)


    # TODO: Multi-Bar Graph (hard) - hint: Try using for loops
    x = np.random.rand(3)
    y = np.random.rand(3)
    z = np.random.rand(3)
    multi_data = [x, y, z]


    # TODO: Histogram Graph
    hist_data = np.random.randn(1000)


    # TODO: Show the image and save it to png file


if __name__ == "__main__":
    main()
