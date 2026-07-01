import numpy as np

# Basis states
zero = np.array([1, 0], dtype=complex)
one = np.array([0, 1], dtype=complex)

# Superposition states
# allows for the creation of superposition states |+> and |-> from the basis states |0> and |1>
#A qubit in state |+⟩ = (1/√2)[1, 1] is not secretly 0 or secretly 1 before measurement. 
# It  has no value. The outcome does not exist until the measurement creates it. 
plus = (zero + one) / np.sqrt(2)
minus = (zero - one) / np.sqrt(2)

# Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard gate
# essentially it is the 2x2 matrix 1/sqrt(2) * [[1, 1], [1, -1]] 
# he Hadamard gate acts as a balanced coin toss: it puts a qubit into a perfectly balanced state
# between 0 and 1. If you measure the qubit, you have a exactly 50% chance of reading a 0 and a 50%
# chance of reading a 1.
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Apply gates to states
# basically we are testing to see if the pauli/other matrices swaps the states as expected.
# note: a 2x2 matrix multiplied by a 2x1 vector results in a 2x1 vector. 

print("X|0> =", X @ zero)
print("X|1> =", X @ one)
print("Z|0> =", Z @ zero)
print("Z|1> =", Z @ one)
print("H|0> =", H @ zero)
print("H|1> =", H @ one)
print("X|+> =", X @ plus)
print("X|-> =", X @ minus)

# Born rule - measurement probabilities
# note; the state inside of measure_probs is the vector u want in the func - 
# e.g measure_probs(zero) - will mean you inputted the vector |0> or [1, 0]into the function.
def measure_probs(state):
# state[0] is the first element of the state vector - e.g if vector is |0> = [1, 0], state[0] = 1 and state[1] = 0.
# btw abs is to take an absolute value. so the abs val of i is 1 
    prob_0 = abs(state[0])**2
    prob_1 = abs(state[1])**2
    return prob_0, prob_1
# in conclusion, born's rule calculates the probability of measuring 0 is |first coefficient|².
# second element — probability of measuring 1 is |second coefficient|².

# asterik seperates the two values of the vector into its own real values.
# essentially the program is calculating the probability of the vector being 1 or 0 with
# the 2 vectors plus and minus bring hadamard states making the probability of measuring 0 and 1 equal to 50% each.
print("\nMeasurement probabilities:")
print("|0>: P(0)={}, P(1)={}".format(*measure_probs(zero)))
print("|1>: P(0)={}, P(1)={}".format(*measure_probs(one)))
print("|+>: P(0)={}, P(1)={}".format(*measure_probs(plus)))
print("|->: P(0)={}, P(1)={}".format(*measure_probs(minus)))

# Simulate actual random measurement
# nothing much else to say tbh, just that the function tracks how many times of measurements it took
# note it will be diffrent mosts of the time because it is random
def simulate_measurement(state, num_trials=1000):
    prob_0 = abs(state[0])**2
    results = np.random.choice([0, 1], size=num_trials, p=[prob_0, 1 - prob_0])
    count_0 = np.sum(results == 0)
    count_1 = np.sum(results == 1)
    print(f"  Measured 0: {count_0} times ({count_0/num_trials:.1%})")
    print(f"  Measured 1: {count_1} times ({count_1/num_trials:.1%})")

print("\nSimulated measurements (1000 trials each):")
print("|0> state:")
simulate_measurement(zero)
print("|+> state:")
simulate_measurement(plus)