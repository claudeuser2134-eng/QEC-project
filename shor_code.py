import numpy as np
import matplotlib.pyplot as plt

zero = np.array([1,0], dtype=complex)
one = np.array([0,1], dtype=complex)

def encode_shor(state):
    alpha = state[0]
    beta = state[1]

    zero3 = np.kron(np.kron(zero, zero), zero) # the qubit zero is |000>
    one3 = np.kron(np.kron(one, one), one) # the qubit one is |111>

    plus_block = (zero3 + one3) / np.sqrt(2) # plus_block is (|000⟩ + |111⟩)/√2
    minus_block = (zero3 - one3) / np.sqrt(2) # minus_block is (|000⟩ - |111⟩)/√2

    zero_L = np.kron(np.kron(plus_block, plus_block), plus_block) # |+>_L = (|000⟩ + |111⟩)/√2 ⊗ (|000⟩ + |111⟩)/√2 ⊗ (|000⟩ + |111⟩)/√2
    one_L = np.kron(np.kron(minus_block, minus_block), minus_block) # |->_L = (|000⟩ - |111⟩)/√2 ⊗ (|000⟩ - |111⟩)/√2 ⊗ (|000⟩ - |111⟩)/√2

    return alpha * zero_L + beta * one_L

encoded = encode_shor(zero) # Encode |0> as |0>_L
print("Encoded |0> length:", len(encoded))
print("Encoded |0> norm:", np.linalg.norm(encoded))


def apply_single_error(state, qubit_index, error_type='X'):
    I = np.eye(2, dtype=complex) # Identity matrix for no error
    X = np.array([[0, 1], [1, 0]], dtype=complex) # X gate for bit flip error
    Z = np.array([[1, 0], [0, -1]], dtype=complex) # Z gate for phase flip error

    if error_type == 'X':
        gate = X
    elif error_type == 'Z':
        gate = Z

    # Apply the error gate to the specified qubit in the 9-qubit state

    ops = [I] * 9 # Start with identity for all qubits
    ops[qubit_index] = gate # Replace the identity with the error gate for the specified qubit.

    result = ops[0]
    for i in range(1, 9):
        result = np.kron(result, ops[i]) # Compute the tensor product of all operators to 
                                         #get the full 9-qubit operator

    return result @ state # Apply the operator to the state and return the new state


encoded = encode_shor(zero) # Encode |0> as |0>_L

errored_z = apply_single_error(encoded, 4, 'Z') # Apply a Z error to the first qubit
errored_x = apply_single_error(encoded, 4, 'X') # Apply an X error to the first qubit

print("X error changes state:", not np.allclose(encoded, errored_x)) 
# Check if the state has changed after X error
print("Z error changes state:", not np.allclose(encoded, errored_z)) 
# Check if the state has changed after Z error

print("Errored states still normalized (X error):", 
      np.isclose(np.linalg.norm(errored_x), 1.0)), np.isclose(np.linalg.norm(errored_z), 1.0)
    # Check if the errored states are still normalized (norm = 1) after X and Z errors


    