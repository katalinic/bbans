"""Demonstrate compression and decompression."""
import numpy as np
import tensorflow as tf

from bbans import encode
from entropy_model import DiagonalGaussianEntropyModel
import mnist_utils
from models.vae import (GaussianVAE, diagonal_gaussian_encoder,
                        bernoulli_decoder, sigmoid_cross_entropy_loss)
from quantiser import MaxEntropyGaussianQuantiser
from rans.rans import init_state

model = GaussianVAE(
    obs_dim=784,
    latent_dim=40,
    encoder=diagonal_gaussian_encoder,
    decoder=bernoulli_decoder,
    decoder_loss=sigmoid_cross_entropy_loss,
    optimiser=tf.train.AdamOptimizer(1e-3),
    seed=0)
model.load_model('./saved_models/')

test_data = mnist_utils.load_test()
test_data = mnist_utils.binarise(test_data)

observation = np.expand_dims(test_data[0], axis=0)
assert observation.shape == (1, 784)

q = MaxEntropyGaussianQuantiser(precision=8)
ent_model = DiagonalGaussianEntropyModel(model.posterior, q, 14)

state = init_state()
latent_to_encode = np.random.randint(0, 1 << 8, 40)

encode(state, latent_to_encode, ent_model, 14, observation)
