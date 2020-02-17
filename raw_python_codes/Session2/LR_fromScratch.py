"""
<Step-by-step guide of Linear Regression>
1. Define a function that computes 'ax+b' with its input 'x'
2. Calculate the difference of prediction (y_hat) and y
3. Define how you are going to update 'a' and 'b', and change their value
4. Iterate above
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

learning_rate = 1e-4
iterations = [10, 20, 100, 1000, 10000]

# x = np.array(
#     [[8.70153760], [3.90825773], [1.89362433], [3.28730045], [7.39333004], [2.98984649], [2.25757240], [9.84450732],
#      [9.94589513], [5.48321616]])
# y = np.array(
#     [[5.64413093], [3.75876583], [3.87233310], [4.40990425], [6.43845020], [4.02827829], [2.26105955], [7.15768995],
#      [6.29097441], [5.19692852]])
x = 5 * np.random.rand(100, 1)
y = 3 * x + 5 * np.random.rand(100, 1)

# plt.scatter(x, y, alpha=1, s=20)
# plt.xlabel("X")
# plt.ylabel("Y")


# outputs predicted equation
def prediction(a, b, x):
    # TODO: return 'x*(transposed)a + b'
    equation = np.dot(x, a.T) + b
    return equation


# By how much are you going to update a and b?
def update_ab(a, b, x, error, lr):
    # Update a
    delta_a = -(lr * (2 / len(error)) * (np.dot(x.T, error)))
    # Update b
    delta_b = -(lr * (2 / len(error)) * np.sum(error))

    return delta_a, delta_b


def caculate_error(a, b, x, y):
    error = y - prediction(a, b, x)
    return error


# calculate error for given number of times, and update a and b
def gradient_descent(x, y, iteration):
    # initial a and b set to a=0 and b=0
    a = np.zeros((1, 1))
    b = np.zeros((1, 1))

    for i in range(iteration):
        print("a: ", a)
        print("b: ", b)
        # TODO: get error
        error = caculate_error(a, b, x, y)
        print("error: ", error)
        # TODO: get 'delta'ed a and b
        delta_a, delta_b = update_ab(a, b, x, error, learning_rate)
        # update a and b
        a -= delta_a
        b -= delta_b

    return a, b


def main():
    fig, ax = plt.subplots(1, 5, figsize=(17, 5))

    for index, iteration in enumerate(iterations):
        final_a, final_b = gradient_descent(x, y, iteration=iteration)
        print("final a:", final_a, "final b:", final_b)

        # Visualise 5 iteration graphs
        y_pred = final_a[0][0] * x + final_b
        ax[index].scatter(x, y)
        ax[index].plot(x, y_pred, color='r')
        ax[index].set_xlabel('X')
        ax[index].set_ylabel('Y')

    fig.savefig("./Image_Output/LR_fromScratch.png")
    # plt.show()


main()
