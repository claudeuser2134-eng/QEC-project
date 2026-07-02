import numpy as np

zero = np.array([1, 0], dtype = complex)
one = np.array([0, 1], dtype = complex)

def encode(state):
    # same thing essentially as bit flip code, uses an encoded state to protect against bit flip errors.
    # sole purpose is so that if one qubit gets flipepd theres still two others
    # turns into 8D vector with 3 qubits. |0> becomes |000> and |1> becomes |111>
    alpha = state[0]
    beta = state[1]

    zero_L = np.zeros(8, dtype=complex)
    zero_L[0] = 1  # |000> is the first basis vector

    one_L = np.zeros(8, dtype=complex)
    one_L[7] = 1   # |111> is the last basis vector

    return alpha * zero_L + beta * one_L

def apply_error(state, qubit_index):
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    if qubit_index == 0:
        error = np.kron(np.kron(X, I), I)
    elif qubit_index == 1:
        error = np.kron(np.kron(I, X), I)
    elif qubit_index == 2:
        error = np.kron(np.kron(I, I), X)
    return error @ state


def measure_syndrome(state):
    probs = np.abs(state)**2
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

def correct(state):
    s1, s2 = measure_syndrome(state)
    if s1 == 1 and s2 == 0:
        return apply_error(state, 0)
    elif s1 == 1 and s2 == 1:
        return apply_error(state, 1)
    elif s1 == 0 and s2 == 1:
        return apply_error(state, 2)
    else:
        return state
    
def run_simulation(error_rate, num_trials=10000): 
    # This function takes two inputs. error_rate is the probability that any single qubit gets flipped 
    # like 0.05 meaning 5% chance.
    fails = 0

    for trial in range(num_trials): # Each time it starts by encoding |0⟩ into |000⟩
        # Simulate a trial with the given error rate
        state = encode(zero)  # Start with |0> encoded as |000>

        for qubit_index in range(3):
            if np.random.rand() < error_rate:
                state = apply_error(state, qubit_index)

# This loops through all three qubits. For each one, it generates a random number
# between 0 and 1 using np.random.random(). 
# If that number is less than error_rate, the qubit gets flipped. If not, nothing happens.


        state = correct(state) # try to correct the state using the syndrome measurement and correction process.

        if not np.allclose(state, encode(zero)):
            fails += 1 
        # It tries to correct whatever happened. Then check whether the corrected state match
        # the original encoded |000⟩ If not, the correction failed and failures goes up by one.

    logical_error_rate = fails / num_trials
    return logical_error_rate


print("Physical error rate -> logical error rate")
print("-" * 45)
for p in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    logical = run_simulation(p)
    print(f"  p = {p:.2f} -> logical = {logical:.4f}")
    # 2f = show 2 decimal places, 4f = show 4 decimal places
    # f just means float.


    # This calls your simulation at eight different error rates and prints the results. 
    # It will take about 10–20 seconds to run because it's doing 80,000 total trials.



# What this program is actually doing overall: Imagine you build a shield to protect a qubit. 
# You want to test how good the shield is. So you throw rocks at it 1000 of times and count 
# how often the shield fails. which is what this program does:

# Protect a qubit by encoding it across three qubits (the shield) - line 62 - 65
# Throw random errors at it (the rocks)—each qubit has a percentage chance of being hit. lines 66-68
# Try to fix any damage using syndrome measurement and correction - line 75
# Check if the fix worked - lines 77-78
# Repeat 10,000 times
# Count what percentage of the time the fix failed

# Then it does this whole thing at eight different throwing intensities
# from gentle (1% chance per qubit) to extreme (50% chance per qubit)
# so you can see exactly when your shield stops being useful.
# The output is data. It answers the question: at what error rate does QEC stop helping? 
# That crossover point is the threshold — the most important number in all of QEC research.


print("\nQuick debug test:")
original = encode(zero)
errored = apply_error(original, 0)
fixed = correct(errored)
print("Single error fix works:", np.allclose(original, fixed))

errored2 = apply_error(apply_error(original, 0), 1)
fixed2 = correct(errored2)
print("Double error fix works:", np.allclose(original, fixed2))