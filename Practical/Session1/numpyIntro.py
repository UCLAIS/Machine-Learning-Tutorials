import numpy as np


def solution():
    x = np.array([[1, 52, 22, 2, 31, 65, 7, 8, 24, 10],
                  [12, 2322, 33, 1, 2, 3, 99, 24, 1, 42],
                  [623, 24, 3, 56, 5, 2, 7, 85, 22, 110],
                  [63, 4, 3, 4, 5, 64, 7, 82, 3, 20],
                  [48, 8, 3, 24, 57, 63, 7, 8, 9, 1032],
                  [33, 64, 0, 24, 5, 6, 72, 832, 3, 10],
                  [12, 242, 2, 11, 52, 63, 32, 8, 96, 2],
                  [13, 223, 52, 4, 35, 62, 7, 8, 9, 10],
                  [19, 2, 3, 149, 15, 6, 172, 2, 2, 11],
                  [34, 23, 32, 24, 54, 63, 1, 5, 92, 7]])

    # First column of x
    firstcol_x = 0

    # Last row of x
    lastrow_x = 0

    # Mean of the first column
    mean_firstcol = 0
    # Mean of the Last row
    mean_lastrow = 0

    # Diagonal components of x
    dia_x = 0
    # Variation of the Diagonal components of x
    var_dia = 0

    return firstcol_x, lastrow_x, mean_firstcol, mean_lastrow, dia_x, var_dia


# function for printing your answers
def print_answer(**kwargs):
    for key in kwargs.keys():
        print(key, ":", kwargs[key])


firstcol_x, lastrow_x, mean_firstcol, mean_lastrow, dia_x, var_dia = solution()

print_answer(firstcol_x=firstcol_x, lastrow_x=lastrow_x, mean_firstcol=mean_firstcol, mean_lastrow=mean_lastrow,
             dia_x=dia_x, var_dia=var_dia)