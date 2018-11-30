# unsuper

Just me plying around with some unsupervised algorithms

Repo structure:
- unsuper/
    * data/ - files for loading data
    * helper/ - helper files
        * encoder_decoder.py - some different encoder and decoder architechtures
        * expm.py - matrix exponential in pytorch
        * losses.py - ELBO loss for variational autoencoders
        * spatial_transformer.py - ST layers for different transformers
        * utility.py - different helper functions
    * models/
        * vae.py - classic variational autoencoder models
        * vitae_ui.py - variational inferred transformational auto encoders with
            semantic inference conditioned on transformational latents
        * vitae_ci.py - variational inferred transformational auto encoders with
            unconditional inference on transformational latents
    * trainer.py - file that does all the optimization work
- main.py - for running experiments

