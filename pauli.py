import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

print("X squared = I:", np.allclose(X @ X, I))
print("Y squared = I:", np.allclose(Y @ Y, I))
print("Z squared = I:", np.allclose(Z @ Z, I))

np.allclose