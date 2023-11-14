import numpy as np
import scipy.special as sp_spec
import scipy.stats as sp_stats
import numpy.random as np_rand
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import seaborn as sns


# Question 1.3.12

def generate_data(N: int, mu: float, tau: float) -> np.ndarray:
    return np_rand.normal(mu, tau, N)


def plot_data(X: np.ndarray, ax: plt.Axes) -> None:
    ax.hist(X, bins=20, density=True)
    ax.set_xlabel('x')
    ax.set_ylabel('p(x)')
    ax.set_title(f'N = {len(X)}')


MU = 1
TAU = 0.5
N = [10, 100, 1000]
Xs = [generate_data(n, MU, TAU) for n in N]

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i in range(len(Xs)):
    plot_data(Xs[i], axs[i])
plt.tight_layout()
plt.savefig('12_data.png')


# Question 1.3.14

def update_tau(x, mu_prime, lambda_prime, a_0, b_0, mu_0, lambda_0):
    E_mu = mu_prime
    E_mu2 = 1 / lambda_prime + mu_prime ** 2

    a_prime = a_0 + x.shape[0] / 2
    b_prime = b_0 + 0.5 * (np.sum(x ** 2) - 2 * np.sum(x) * E_mu +
                           x.shape[0] * E_mu2 + lambda_0*(E_mu2 + mu_0 ** 2 - 2 * E_mu * mu_0))

    return a_prime, b_prime


def update_mu(x, a_prime, b_prime, mu_0, lambda_0):
    E_tau = a_prime / b_prime
    mu_prime = (lambda_0 * mu_0 + np.sum(x)) / (lambda_0 + x.shape[0])
    lambda_prime = (lambda_0 + x.shape[0])*E_tau

    return mu_prime, lambda_prime
