import numpy as np


def solution():
    prime = [2, 3, 5, 7, 11]

    matrix = [
        # TODO: Try making your own matrix by using the list 'prime'
        # TODO: Try doing sth awesome. Don't just write tons of numbers
              ]
    # TODO: make it as a numpy array
    matrix = np.array(matrix)
    # TODO: What is Diagonal of the above matrix?
    matrix_dia = np.diagonal(matrix)

    # TODO: What are the sum and mean of the diagonal components?
    dia_sum = np.sum(matrix_dia)
    dia_mean = np.mean(matrix_dia)

    return matrix, dia_sum, dia_mean


# This function is for printing your answers
def print_answer(**kwargs):
    for key in kwargs.keys():
        print(key, ":", kwargs[key])


matrix, dia_sum, dia_mean = solution()

print_answer(matrix=matrix, dia_sum=dia_sum, dia_mean=dia_mean)