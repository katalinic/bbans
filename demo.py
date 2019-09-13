"""Demonstrate compression and decompression."""
import numpy as np
import tensorflow as tf

from bbans import encode, decode
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

prior_prec = 8
latent_prec = 14

q = MaxEntropyGaussianQuantiser(precision=prior_prec)
ent_model = DiagonalGaussianEntropyModel(model.posterior, q, latent_prec)

_state = np.random.randint(0, 2, 400)
state = init_state(_state)
latent_to_encode = np.random.randint(0, 1 << prior_prec, 40)

state = encode(state, latent_to_encode, ent_model, latent_prec, observation)

state, symbol = decode(state, ent_model, latent_prec, observation)
