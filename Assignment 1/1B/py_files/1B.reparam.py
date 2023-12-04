def reparameterization(self, mean, var):
    # insert your code here
    std = torch.sqrt(var + 1e-10)
    eps = torch.randn_like(std)
    z = mean + std * eps

    return z
