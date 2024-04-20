import numpy as np
import matplotlib.pyplot as plt


# Create likelihood matrix
def create_likelihood_matrix(size, ambiguity):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distance = abs(i - j)
            matrix[i, j] = np.exp(-0.5 * (distance / ambiguity) ** 2)
    norm_matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return norm_matrix

# Helper function to prevent log(0)
def log_stable(x, minval=1e-30):
    return np.log(np.maximum(x, minval))

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Get approximate posterior
def infer_states(observation_index, likelihood_matrix, prior):
    log_likelihood = log_stable(likelihood_matrix[observation_index, :])
    log_prior = log_stable(prior)
    qs = softmax(log_likelihood + log_prior)
    return qs

# KL divergence
def kl_divergence(p, q):
    return (log_stable(p) - log_stable(q)).dot(p)

# Construct prior
def get_prior(prior_state, size, ambiguity):
    prior = np.zeros(size)
    for i in range(size):
        distance = abs(i - prior_state)
        prior[i] = np.exp(-0.5 * (distance / ambiguity) ** 2)
    norm_prior = prior / prior.sum()
    return norm_prior

# Visualization of likelihood
def plot_likelihood(matrix):
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Likelihood Matrix')
    plt.xlabel('Observations')
    plt.ylabel('States')
    plt.show()

# Visualization of priors / posterior
def plot_belief(belief, title):
    plt.figure(figsize=(5, 2))
    plt.plot(belief, marker = None, linestyle='-', color='black')
    plt.title(f'{title}')
    plt.xlabel('State')
    plt.grid(True)
    plt.show()

def inference(size, decay1, decay2, likelihood, prior, obs):
    qs = infer_states(obs, likelihood, prior)
    dkl = kl_divergence(qs, likelihood[obs, :])
    evidence = log_stable(prior)
    F = dkl - evidence
    return qs, F
