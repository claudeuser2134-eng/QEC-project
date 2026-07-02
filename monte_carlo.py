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
    elif s1 == 0 and s2 == 1:
        return apply_error(state, 1)
    elif s1 == 1 and s2 == 1:
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


        state = correct(state)

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

    # This calls your simulation at eight different error rates and prints the results. 
    # It will take about 10–20 seconds to run because it's doing 80,000 total trials.