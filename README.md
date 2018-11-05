# unsuper

Just me plying around with some unsupervised algorithms

Repo structure:
- unsuper/
    * data/ - files for loading data
    * helper/ - helper files
    * models/
        * vae_*.py - classic variational autoencoder models
        * vitae_*.py - variational inferred transformational auto encoders with
        semantic inference conditioned on transformational latents
        * vitae2_*.py - variational inferred transformational auto encoders with
        no conditional inference
    * trainer.py - file that does all the optimization work
- main.py - for running experiments

