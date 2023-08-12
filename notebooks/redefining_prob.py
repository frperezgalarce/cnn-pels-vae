from scipy.stats import norm
import numpy as np

# Define the Gaussian mixture components
means = [0, 5]
std_devs = [1, 1]
weights = [0.5, 0.5]

# Define the original mixture of Gaussians
def p(x):
    return sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in zip(weights, means, std_devs))

# Define the modified distribution q(x)
b = 0.5
def q(x):
    return p(x)**b / Z

# Estimate the normalization constant Z using numerical integration
from scipy.integrate import quad
Z, _ = quad(lambda x: p(x)**b, -np.inf, np.inf)

# Now you can use q(x) in a sampling method like rejection sampling or Metropolis-Hastings
