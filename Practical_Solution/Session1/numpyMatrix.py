import numpy as np

array1 = np.array([[3,2,1], [14,53,2], [1,6,9]])
print(array1)

# TODO: Transpose array1
transposed = np.transpose(array1)
print(transposed, 'is a transposed Matrix')

# TODO: Dot product of array1 and its transposed Matrix
power = np.dot(transposed, array1)
print(power, 'is array1 (dot) array1^T')

# TODO: Do elementwise multiplication of array1 and its transposed Matrix
elementwise_prod = transposed * array1
print(elementwise_prod, 'is elementwise multiplication of array1 and its transposed Matrix')

array2 = np.array([[3,10],[5,74]])

# TODO: Make an inverse matrix of array2
inverse_array2 = np.linalg.inv(array2)
print(inverse_array2, 'is an inverse matrix of array2')

# TODO: Get diagonal components of array2
diagonal = np.diagonal(array2)
print(diagonal, 'is a diagonal components of array2')

# TODO: Dot product of array2 and its inverse matrix
producted = np.dot(array2, inverse_array2)
print(producted, 'is a dot product of array2 and its inverse matrix')