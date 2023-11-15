from scipy.stats import gamma, norm

# prior parameters
mu_0 = 0
lambda_0 = 1
a_0 = 1
b_0 = 1


def compute_elbo(D, a_0, b_0, mu_0, lambda_0, a_N, b_N, mu_N, lambda_N):
    # given the prior and posterior parameters together with the data,
    # compute ELBO here
    N = len(D)

    return elbo


def CAVI(D, a_0, b_0, mu_0, lambda_0):
    # make an initial guess for the expected value of tau
    initial_guess_exp_tau = 1

    N = len(D)
    x_mean = np.mean(D)
    x_2_sum = np.sum(D**2)

    # Constants
    a_N = a_0 + N / 2
    mu_N = (lambda_0 * mu_0 + N * x_mean) / (lambda_0 + N)
    E_mu = mu_N

    # Variational parameters
    b_N = b_0
    lambda_N = lambda_0

    # ELBO
    elbos = []

    # CAVI iterations ...
    for i in range(100):
        # update the values for the variational parameters
        E_tau = a_N / b_N
        E_mu_2 = 1 / lambda_N + mu_N**2

        lambda_N = (lambda_0 + N) * E_tau
        b_N = b_0 + 1 / 2 * (x_2_sum + N*E_mu_2 - 2*N*E_mu *
                             x_mean + lambda_0*(E_mu_2 - 2*E_mu*mu_0 + mu_0**2))
        # save ELBO for each iteration, plot them afterwards to show convergence
        elbos.append(compute_elbo(D, a_0, b_0, mu_0,
                     lambda_0, a_N, b_N, mu_N, lambda_N))

    return a_N, b_N, mu_N, lambda_N, elbos
