def loss_function(x, theta, mean, log_var):  # should return the loss function (- ELBO)
    # insert your code here
    # Approximation of the expected log-likelihood
    exp_log_likelihood = torch.sum(
        x * torch.log(theta) + (1 - x) * torch.log(1 - theta), dim=1)

    # KL divergence between the variational distribution and the prior
    kl_divergence = -0.5 * \
        torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)

    # ELBO
    loss = torch.mean(kl_divergence - exp_log_likelihood)

    return loss
