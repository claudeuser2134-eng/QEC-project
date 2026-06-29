import numpy as np

zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

# Two-qubit basis states using tensor product
# np.kron computes the tensor product (when 2x1 matrices does something to get to get a 4x1 matrix)
# Two qubits = 4-dimensional vector
two_zeros = np.kron(zero, zero)    # |00>
zero_one = np.kron(zero, one)      # |01>
one_zero = np.kron(one, zero)      # |10>
two_ones = np.kron(one, one)       # |11>

print("Two-qubit basis states:")
print("|00> =", two_zeros)
print("|01> =", zero_one)
print("|10> =", one_zero)
print("|11> =", two_ones)

# CNOT gate allows it to act on two qubits at once
# First qubit = control, second qubit = target
# If control is |0>, do nothing to the target
# If control is |1>, flip the target (apply X gate to it)
# This gate is essential for error correction because it lets us check for errors without 
# measuring the data qubit directly whcih is important to QEC
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

# CNOT|00⟩ — first qubit is 0, so nothing happens → should stay |00⟩
# CNOT|01⟩ — first qubit is 0, so nothing happens → should stay |01⟩
# CNOT|10⟩ — first qubit is 1, so second qubit flips → should become |11⟩
# CNOT|11⟩ — first qubit is 1, so second qubit flips → should become |10⟩
print("\nCNOT gate applied to each basis state:")
print("CNOT|00> =", CNOT @ two_zeros)
print("CNOT|01> =", CNOT @ zero_one)
print("CNOT|10> =", CNOT @ one_zero)
print("CNOT|11> =", CNOT @ two_ones)
# remember 4x4 matirx multiplied with a 4x1 vector will give a 4x1 vector as output

# Create the |+> state for the first qubit
plus = (zero + one) / np.sqrt(2)

# Two-qubit input: first qubit in |+>, second in |0>
# np.kron computes the tensor product
input_state = np.kron(plus, zero)
print("\nInput state |+>|0> =", input_state)

# Apply CNOT to create a Bell state
# A Bell state is two qubits entangled so they always agree (both 0 or both 1) 
# but which outcome occurs is random.
bell_state = CNOT @ input_state
print("After CNOT (Bell state) =", bell_state)

# Check measurement probabilities
# essentially the first number is the control and the second number is the target
# CNOT|00⟩ = |00⟩   first digit 0→0, second digit 0→0 (control is 0, do nothing)
# CNOT|01⟩ = |01⟩   first digit 0→0, second digit 1→1 (control is 0, do nothing)
# CNOT|10⟩ = |11⟩   first digit 1→1, second digit 0→1 (control is 1, flip target)
# CNOT|11⟩ = |10⟩   first digit 1→1, second digit 1→0 (control is 1, flip target)
# For this particular bell, state only 00 and 11 are possible outcomes, 
# so the probabilities of measuring 01 or 10 should be 0.
probs = np.abs(bell_state)**2
labels = ["|00>", "|01>", "|10>", "|11>"]
print("\nBell state measurement probabilities:")
for i in range(4):
    print(f"  {labels[i]}: {probs[i]:.1%}")