import numpy as np
import scipy.special as sp_spec
import scipy.stats as sp_stats
import numpy.random as np_rand
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import trange
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

def update_b_N(x, mu_N, lambda_N, b_0, mu_0, lambda_0):
    E_mu = mu_N
    E_mu2 = 1 / lambda_N + mu_N ** 2

    b_N = b_0 + 0.5 * (np.sum(x ** 2) - 2 * np.sum(x) * E_mu +
                       x.shape[0] * E_mu2 + lambda_0*(E_mu2 + mu_0 ** 2 - 2 * E_mu * mu_0))

    return b_N


def update_lambda_N(x, a_N, b_N, lambda_0):
    E_tau = a_N / b_N
    lambda_N = (lambda_0 + x.shape[0])*E_tau

    return lambda_N


def vi_alg(x, a_0, b_0, mu_0, lambda_0, iter=20):
    N = x.shape[0]

    # Constants
    a_N = a_0 + N/2
    mu_N = (lambda_0 * mu_0 + np.sum(x)) / (lambda_0 + N)

    # Variables
    b_N = b_0
    lambda_N = lambda_0

    # Lists for plotting
    b_Ns = np.zeros(iter+1)
    lambda_Ns = np.zeros(iter+1)

    b_Ns[0] = b_N
    lambda_Ns[0] = lambda_N

    for i in trange(iter):
        b_Ns[i+1] = update_b_N(x, mu_N, lambda_Ns[i], b_0, mu_0, lambda_0)
        lambda_Ns[i+1] = update_lambda_N(x, a_N, b_Ns[i], lambda_0)

        a_True, b_True, mu_True, lambda_True = true_posterior(
            x, a_0, b_0, mu_0, lambda_0)

    print('a_N =', a_N)
    print('a_True =', a_True)
    print('b_N =', b_Ns[-1])
    print('b_True =', b_True)
    print('mu_N =', mu_N)
    print('mu_True =', mu_True)
    print('lambda_N =', lambda_Ns[-1])
    print('lambda_True =', lambda_True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(b_Ns)
    axs[0].axhline(b_True, color='r', linestyle='--')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('b_N')
    axs[0].set_title('b_N')
    axs[1].plot(lambda_Ns)
    axs[1].axhline(lambda_True, color='r', linestyle='--')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('lambda_N')
    axs[1].set_title('lambda_N')
    plt.tight_layout()
    plt.show()
    # plt.savefig('14_vi.png')


# Question 1.3.15

def true_posterior(x, a_0, b_0, mu_0, lambda_0):
    mu_N = (lambda_0 * mu_0 + np.sum(x)) / (lambda_0 + x.shape[0])
    lambda_N = lambda_0 + x.shape[0]
    a_N = a_0 + x.shape[0]/2
    b_N = b_0 + 0.5 * (np.sum(x ** 2) + lambda_0 *
                       mu_0 ** 2 - lambda_N * mu_N ** 2)
    return a_N, b_N, mu_N, lambda_N


a_0 = 1
b_0 = 1
mu_0 = 1
lambda_0 = 12

vi_alg(Xs[-1], a_0, b_0, mu_0, lambda_0)
