import jetm as jm

def multiply_combinations(x, y):
    matrix = []
    for i in range(x.shape[0]):
        matrix.append([])
        for j in range(y.shape[0]):
            matrix[i].append(x[i, 0] * y[j, 0])
    return jm.matrix(matrix)

def multiply_across(m, v):
    if m.shape[0] != v.shape[0]:
        raise Exception("Matrix and vector rows count do not match")
    matrix = []
    for i in range(v.shape[0]):
        matrix.append([])
        for j in range(m.shape[1]):
            matrix[i].append(m[i, j] * v[i, 0])
    return jm.matrix(matrix)

