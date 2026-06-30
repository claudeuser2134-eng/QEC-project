
from dbm import error

import numpy as np

# the objective is to have one qubit that you protect from errors. One qubit alone is vulnerable 
# and if it gets flipped theres have no way to detect that.
# The solution is to spread the information across three qubits instead of one
# Not by copying as (the no-cloning theorem forbids that), but by encoding.

# Single qubit basis states
zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

# Encode a single qubit into three qubits
# |0> becomes |000>
# |1> becomes |111>
def encode(state):
    # If the qubit is |0>, encode as |000>
    # If the qubit is |1>, encode as |111>
    # For a superposition a|0> + b|1>, encode as a|000> + b|111>
    alpha = state[0]  # coefficient of |0>
    beta = state[1]   # coefficient of |1>
    # This reads the two coefficients of the qubit. 
    # For |0⟩ = [1, 0], alpha is 1 and beta is 0. For |+⟩, both are 1/√2.

    # 1 qubit in 2D space, 2 qubits in 4D space, 3 qubits in 8D space (IMPORTANT)

    # |000> as an 8-dimensional vector
    zero_L = np.zeros(8, dtype=complex)
    zero_L[0] = 1  # |000> is the first basis vector with it being [1,0,0,0,0,0,0,0]

    # |111> as an 8-dimensional vector
    one_L = np.zeros(8, dtype=complex)
    one_L[7] = 1   # |111> is the last basis vector. with it being [0,0,0,0,0,0,0,1]

    # Encoded state: alpha|000> + beta|111>
    encoded = alpha * zero_L + beta * one_L
    return encoded

# Test encoding
encoded_zero = encode(zero)
encoded_one = encode(one)
print("Encoded |0>:", encoded_zero)
print("Encoded |1>:", encoded_one)

# Encode a superposition state
plus = (zero + one) / np.sqrt(2)
encoded_plus = encode(plus)
print("Encoded |+>:", encoded_plus)

# If the original qubit is |0⟩, store it as |000⟩ — all three qubits set to 0.
# If the original qubit is |1⟩, store it as |111⟩ — all three qubits set to 1.
# If the original qubit is a superposition like |+⟩ = (1/√2)|0⟩ + (1/√2)|1⟩, 
# we can store it as (1/√2)|000⟩ + (1/√2)|111⟩.


# Apply a bit-flip error (X gate) to one specific qubit
# qubit_index: 0, 1, or 2 (which of the three qubits to flip)
def apply_error(state, qubit_index):
    I = np.eye(2, dtype=complex)  # Identity matrix for no flip
    X = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli-X gate (bit-flip)

    if qubit_index == 0:
        # Apply X to the first qubit
        error = np.kron(np.kron(X, I), I)
    elif qubit_index == 1:
        # Apply X to the second qubit
        error = np.kron(np.kron(I, X), I)
    elif qubit_index == 2:
        error = np.kron(np.kron(I, I), X)

    return error @ state

# Test: flip qubit 0 of encoded |0> = |000>
errored = apply_error(encoded_zero, 0)
print("\nAfter flipping qubit 0 of |000>:", errored)
# predicted result is |100> = [0, 0, 0, 0, 1, 0, 0, 0]

errored1 = apply_error(encoded_zero, 1)
print("\nAfter flipping qubit 1 of |000>:", errored1)
# predicted result is |010> = [0, 0, 0, 1, 0, 0, 0, 0]

errored2 = apply_error(encoded_zero, 2)
print("\nAfter flipping qubit 2 of |000>:", errored2)
# predicted result is |001> = [0, 0, 1, 0, 0, 0, 0, 0]



def measure_syndrome(state):
    probs = np.abs(state)**2

    s1 = 0
    s2 = 0

    for i in range(8):
        if probs[i] > 0.00001:
            q0 = (i >> 2) & 1  # first qubit
            q1 = (i >> 1) & 1  # second qubit
            q2 = i & 1         # third qubit

            s1 = q0 ^ q1  # 
            s2 = q1 ^ q2  # parity of last two qubits

    return s1, s2

# Test syndrome on each error case
print("\nSyndrome measurements:")
print("No error on |000>:", measure_syndrome(encoded_zero))
print("Error on qubit 0:", measure_syndrome(apply_error(encoded_zero, 0)))
print("Error on qubit 1:", measure_syndrome(apply_error(encoded_zero, 1)))
print("Error on qubit 2:", measure_syndrome(apply_error(encoded_zero, 2)))

def correct(state):
    s1, s2 = measure_syndrome(state)

    if s1 == 0 and s2 == 0:
        print(" Syndrome (1, 0) error on qubit 0, correcting...")
        return apply_error(state, 0)  # No error, return original state + applying X again to undo
    elif s1 == 1 and s2 == 1:
        print(" Syndrome (1, 1) error on qubit 1, correcting...")
        return apply_error(state, 1)
    elif s1 == 0 and s2 == 1:
        print(" Syndrome (0, 1) error on qubit 2, correcting...")
        return apply_error(state, 2)
    else:
        print(" Syndrome (0, 0) no error detected, returning original state")
        return state  # No error detected, return original state
    
# Full state: encode -> introduce error -> detect error -> correct error
print("\nFull error correction cycle:")
original = encode(zero)
corrupted = apply_error(original, 1)  # Introduce error on qubit 1
recovered = correct(corrupted)  # Detect and correct the error
print("Original state:", original)
print("Corrupted state:", corrupted)
print("Recovered state:", recovered)
print("Recovery successful:", np.allclose(original, recovered))
