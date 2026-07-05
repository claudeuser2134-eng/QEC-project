import numpy as np
import matplotlib.pyplot as plt

zero = np.array([1,0], dtype=complex)
one = np.array([0,1], dtype=complex)

def encode(state):
    alpha = state[0]
    beta = state[1]
    zero_L = np.zeros(8, dtype=complex)
    zero_L[0] = 1  # |000> is the first basis vector
    one_L = np.zeros(8, dtype=complex)
    one_L[7] = 1

    return alpha * zero_L + beta * one_L

def apply_error(state, qubit_index):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
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
    
def run_simulation(error_rate, num_trials=10000):
    fails = 0
    for trial in range(num_trials):
        state = encode(zero)  # Start with |0> encoded as |000>
        for qubit in range(3):
            if np.random.random() < error_rate:
                state = apply_error(state, qubit)
        state = correct_state(state)
        if not np.allclose(state, encode(zero)):
            fails += 1
    return fails / num_trials

# list of physical error rates
# using more values to make sure the graph runs more smoothly
error_rates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# empty lists will be used to store the logical error rate per physical error rate
logical_rates = []

# Run the simulation at each error rate and store the result
# This takes about 30 seconds because it runs 10,000 trials at each of 11 error rates
print("Running simulations...")
for p in error_rates:
    # run_simulation returns the fraction of trials that failed
    logical = run_simulation(p)
    # add it to our list
    logical_rates.append(logical)
    # print progress so you know it's working
    print(f"  p = {p:.2f} done -> logical = {logical:.4f}")

print("All simulations complete. Generating graph...")
# generating graph

# creating a new figure/diagram
# the whole figutre will be approx 8 by 6 in inches
fig, ax = plt.subplots(figsize = (8, 6))

# plotting the monte carlo results as the blue lines and dots
# 'o-' means circle is connected by lines
# label is what appears in the legend
ax.plot(error_rates, logical_rates,'o-', color='blue', label='3-qubit bit-flip code')

# plot the diagonal line p = p
# this showns like an average or what happens when you dont do anything 
# ( the logical error rate equals the physical rate)

# the line below the diagonal is the code helping 
# the line above the diagonal is the code making things worse or more errors
ax.plot([0, 0.5], [0, 0.5], '--', color='gray', label='No correction (p = p)')


# plot the theortetical graph prediction of the 3^2
# expected logical rate plots for a perfect 3 qubit code
# it stems from the probbility of 2 or more errors occuring simultaneously
theory_p = np.linspace(0, 0.5, 100)  # 100 points between 0 and 0.5
theory_logical = 3 * theory_p**2      # formula: 3p^2
ax.plot(theory_p, theory_logical, '-', color='red', label='Theory: 3p²')

#label the x-axis
ax.set_xlabel('Physical error rate (p)', fontsize=12)

#labeling the y-axis
ax.set_ylabel('Logical error rate', fontsize=12)

# adding the key or legend
ax.legend(loc='upper left', fontsize=10)

ax.grid(True, alpha=0.3)

#setting limits for the graph
ax.set_xlim(0, 0.52)
ax.set_ylim(0, 0.8)

# Save the graph as an image file in your project folder
# dpi=150 makes it high resolution
plt.savefig('error_rate_plot.png', dpi=150, bbox_inches='tight')
# bbox_inches='tight' removes extra white space around the edges

# Show the graph on screen
plt.show()

# Print a message confirming the image was saved
print("Graph saved as error_rate_plot.png")

