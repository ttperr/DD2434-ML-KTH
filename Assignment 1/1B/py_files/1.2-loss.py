def loss_function(x, theta, mean, log_var):  # should return the loss function (- ELBO)
    # insert your code here
    # expected log-likelihood
    recon_loss = -torch.sum(x * torch.log(theta + 1e-10) +
                            (1 - x) * torch.log(1 - theta + 1e-10))

    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    loss = recon_loss + kl_div

    return loss
