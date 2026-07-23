import numpy as np

zero = np.array([0,1], dtype=complex)
one = np.array([1,0], dtype=complex)

def encode(state):
    alpha = state[0]
    beta = state[1]

    zero_L = np.array([1,0,0,0,0,0,0,0], dtype=complex)  # |000> is the first basis vector
    one_L = np.array([0,0,0,0,0,0,0,1], dtype=complex)   # |111> is the last basis vector

    return alpha * zero_L + beta * one_L

def apply_error(state, qubit_index):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X gate (bit-flip)

    if qubit_index == 0:
        error = np.kron(np.kron(X, I), I)
    elif qubit_index == 1:
        error = np.kron(np.kron(I, X), I)
    elif qubit_index == 2:
        error = np.kron(np.kron(I, I), X)
    return error @ state


def measure_syndrome(state):
    s1 = 0
    s2 = 0

    probs = np.abs(state)**2
    for i in range(8):
        if probs[i] > 0.00001:
            q0 = (i >> 2) & 1
            q1 = (i >> 1) & 1
            q2 = i & 1

            s1 = q0 ^ q1
            s2 = q1 ^ q2
    return s1, s2


def correct_state(state):
    s1, s2 = measure_syndrome(state)
    if s1 == 1 and s2 == 0:
        return apply_error(state, 0)
    elif s1 == 1 and s2 == 1:
        return apply_error(state, 1)
    elif s1 == 0 and s2 == 1:
        return apply_error(state, 2)
    else:
        return state
    

