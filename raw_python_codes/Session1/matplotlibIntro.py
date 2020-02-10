import numpy as np
import matplotlib.pyplot as plt

def main():
    # TODO: Plot y=x and y=x^2 graphs
    x = np.arange(30)
    print(x)
    fig, ax = plt.subplots()

    ax.plot(
        x, x, label='y=x',
        marker='o',
        color='blue',
        linestyle=':'
    )
    ax.plot(
        x, x ** 2, label='y=x^2',
        marker='^',
        color='red',
        linestyle='--'
    )

    # TODO: Set labels, limits of range of axis, and make legend in good position
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)

    ax.legend(
        loc='upper left',
        shadow=True,
        fancybox=True,
        borderpad=1    # border padding of the legend box
    )

    # TODO: Show the image you created and save it to png file
    plt.show()
    # plt.savefig("sample.png")

if __name__ == "__main__":
    main()
