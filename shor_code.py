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
    for i in range(1, 9): # creates the full 512 x 512 matrix by chaining tensor products 
                          # via np.kron instead of writing it out 9 times 
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


# in the 3 qubit code we could use syndrome bits: looking at which position in the 8D vector
# had non-zero probability, then extracting bits with shifting.

# However as we approach larger values like 512 matrix (or 9 qubits) it gets harder so we have
# to switch to a new concept called stablizer measurements.

# a stabilizer is an operater(matrix) that leaves the correct codework unchanged


def measure_stabilizer(state, qubit_indices, gate_type='Z'):
    I = np.eye(2, dtype=complex) # Identity matrix for no error
    X= np.array([[0, 1], [1, 0]], dtype=complex) # X gate for bit flip error
    Z = np.array([[1, 0], [0, -1]], dtype=complex) # Z gate for phase flip error

    if gate_type == 'Z':
        gate = Z
    else:
        gate = X

    ops = [I] * 9 # Start with identity for all qubits
    for idx in qubit_indices:
        ops[idx] = gate # place Z or X gates at the specific qubit positions you want to compare. 
        # For Z₀Z₁, qubit_indices would be [0, 1], so positions 0 and 1 get Z 
        # while the other 7 stay as I.

    stabilizer = ops[0]
    for i in range(1, 9): # creates the full 512 x 512 matrix by chaining tensor products 
                          # via np.kron to make it faster
        stabilizer = np.kron(stabilizer, ops[i]) # Compute the tensor product of all operators to 
                                                   # get the full 9-qubit stabilizer operator
    
    expectation = np.real(state.conj() @ stabilizer @ state)
     # state.conj() is the conjugate of the state vector (flips the sign of any imaginary parts)
     # Calculate the expectation value of the stabilizer
    return int(round((1 - expectation) / 2)) # Return 0 if expectation is +1, 
                                             # and 1 if expectation is -1

def measure_shor_syndrome(state):
    bit_flip_syndrome = []

    for block in range(3):
        offset = block * 3
        s1 = measure_stabilizer(state, [offset, offset + 1], 'Z')
        s2 = measure_stabilizer(state, [offset + 1, offset + 2], 'Z')
        bit_flip_syndrome.append((s1, s2))


    phase_s1 = measure_stabilizer(state, [0,1,2,3,4,5], 'X')
    phase_s2 = measure_stabilizer(state, [3,4,5,6,7,8], 'X')

    return bit_flip_syndrome, (phase_s1, phase_s2)
        
def correct_shorcode(state):
    bit_flip_syndrome, phase_syndrome = measure_shor_syndrome(state)
     # bit_flip_syndromes is a list of three tuples like [(0,0), (1,0), (0,0)]
     # the phase syndrome is only one tuple of (0,1)

    for block in range(3): # so we dont have to repeat it 3 times as it is already repeated
        s1, s2 = bit_flip_syndrome[block] 
        offset = block * 3 # formula to find the actual qubit
        if s1 == 1 and s2 == 0:
            state = apply_single_error(state, offset + 0, 'X') # using the formula provided 
        elif s1 == 1 and s2 == 1:
            state = apply_single_error(state, offset + 1, 'X')
        elif s1 == 0 and s2 == 1:
            state = apply_single_error(state, offset + 2, 'X')

    phase1, phase2 = phase_syndrome
    if phase1 == 1 and phase2 == 0:
        state = apply_single_error(state, 0, 'Z')
    elif phase1 == 1 and phase2 == 1:
        state = apply_single_error(state, 3, 'Z')
    elif phase1 == 0 and phase2 == 1:
        state = apply_single_error(state, 6, 'Z')

    return state


# testing if the shor code works

encoded = encode_shor(zero)

print("=== Bit-flip error tests ===")
for q in range(9):
    errored = apply_single_error(encoded, q, 'X')
    fixed = correct_shorcode(errored)
    print(f"X error on qubit {q}: Recovery = {np.allclose(encoded, fixed)}")

print("\n=== Phase-flip error tests ===")
for q in range(9):
    errored = apply_single_error(encoded, q, 'Z')
    fixed = correct_shorcode(errored)
    print(f"Z error on qubit {q}: Recovery = {np.allclose(encoded, fixed)}")

print("\n=== Y error tests ===")
for q in range(9):
    errored = apply_single_error(encoded, q, 'X')
    errored = apply_single_error(errored, q, 'Z')
    fixed = correct_shorcode(errored)
    overlap = abs(np.conj(encoded) @ fixed)
    print(f"Y error on qubit {q}: Recovery = {np.isclose(overlap, 1.0)}")
    

