import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

for name, matrix in [("X", X), ("Y", Y), ("Z", Z)]:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print(f"Eigenvalues of {name}: {eigenvalues}")
    print(f"Eigenvectors of {name}:")
    print(eigenvectors)
    print()