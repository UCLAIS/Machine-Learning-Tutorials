import numpy as np

x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

def solution(arr):
    # make x as numpy array
    array1 = np.array(arr)

    # 3x3 matrix filled with 0s
    array2 = np.zeros((3, 3))

    # 2x5 matrix filled with 1s
    array3 = np.ones((2, 5))

    # 5×3 matrix filled with trash (uninitialised) values
    array4 = np.empty((5, 3))

    # single matrix filled with number from 0 to 9
    array5 = np.arange(10)

    return array1, array2, array3, array4, array5


def print_answer(**kwargs):
    for key in kwargs.keys():
        print(key, ":", kwargs[key])


array1, array2, array3, array4, array5 = solution(x)

print_answer(array1=array1, array2=array2, array3=array3, array4=array4, array5=array5)
