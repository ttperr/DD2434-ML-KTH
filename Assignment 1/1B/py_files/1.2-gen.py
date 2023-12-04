with torch.no_grad():
    # insert your code here to create images from noise (it is enough to create theta value for each pixel)
    #
    #
    # generated_images = ....  # should be a matrix ( batch_size-by-x_dim )
    generated_images = torch.round(
        model.Decoder(torch.randn(batch_size, latent_dim)))
