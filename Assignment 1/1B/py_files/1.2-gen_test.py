model.eval()
# below we get decoder outputs for test data
with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim)
        # insert your code below to generate theta from x

        # Pass the test images through the encoder and decoder
        mean, log_var = model.Encoder(x)
        # reparameterize to get latent variable
        z = model.reparameterization(mean, torch.exp(log_var))
        # decode the latent variable to get reconstructed image
        theta = model.Decoder(z)
