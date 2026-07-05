import numpy as np

zero = np.array([1,0], dtype=complex)
one = np.array([0,1], dtype=complex)

plus = (zero + one) / np.sqrt(2)
minus = (zero - one) / np.sqrt(2)

H = np.array([[1,1], [1,-1]], dtype=complex) / np.sqrt(2)

def encode_phase(state):
    alpha = state[0]
    beta = state[1]

    Plus_L = np.kron(np.kron(plus, plus), plus)
    Minus_L = np.kron(np.kron(minus, minus), minus)

    return alpha * Plus_L + beta * Minus_L


def apply_phase_error(state, qubit_index):
    I = np.eye(2, dtype=complex)
    Z = np.array([[1,0], [0,-1]], dtype=complex)
    if qubit_index == 0:
        error = np.kron(np.kron(Z, I), I)
    elif qubit_index == 1:
        error = np.kron(np.kron(I, Z), I)
    elif qubit_index == 2: 
        error = np.kron(np.kron(I, I), Z)
    return error @ state

def measure_phase_syndrome(state):
    H3 = np.kron(np.kron(H, H), H)
    transformed = H3 @ state
    probs = np.abs(transformed)**2

    s1 = 0
    s2 = 0

    for i in range(8):
        if probs[i] > 0.0001:
            q0 = (i >> 2) & 1
            q1 = (i >> 1) & 1
            q2 = i & 1
            s1 = q0 ^ q1
            s2 = q1 ^ q2
    return s1, s2


def correct_phase(state):
    s1, s2 = measure_phase_syndrome(state)
    if s1 == 1 and s2 == 0:
        return apply_phase_error(state, 0)
    elif s1 == 1 and s2 == 1:
        return apply_phase_error(state, 1)
    elif s1 == 0 and s2 == 1:
        return apply_phase_error(state, 2)
    else:
        return state
    

print("Phase flip code tests:")
original = encode_phase(zero)
for q in range(3):
    errored = apply_phase_error(original, q)
    fixed = correct_phase(errored)
    print(f"Error on qubit {q}: Recovery = {np.allclose(original, fixed)}")
