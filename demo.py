import numpy as np
import tensorflow as tf

from bbans import bbans_encode, bbans_decode
from entropy_model import DiagonalGaussianEntropyModel, BernoulliEntropyModel
import mnist_utils
from models.vae import (GaussianVAE, diagonal_gaussian_encoder,
                        bernoulli_decoder, sigmoid_cross_entropy_loss)
from quantiser import MaxEntropyGaussianQuantiser
from rans.rans import init_state, flatten_state, unflatten_state

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

prior_prec = 8
latent_prec = 14
obs_prec = 12


def _prior_dist(*args):
    return np.zeros(40), np.ones(40)


q = MaxEntropyGaussianQuantiser(precision=prior_prec)
prior_ent_model = DiagonalGaussianEntropyModel(_prior_dist, q, prior_prec)
posterior_ent_model = DiagonalGaussianEntropyModel(
    model.posterior, q, latent_prec)
obs_ent_model = BernoulliEntropyModel(model.generate, None, obs_prec)

rng = np.random.RandomState(0)
init_bits = rng.randint(low=1 << 16, high=1 << 31, size=20, dtype=np.uint32)
state = init_state()
state = unflatten_state(init_bits)

rng.shuffle(test_data)
num_imgs = 200
for i, image in enumerate(test_data[:num_imgs]):
    state = bbans_encode(state,
                         np.expand_dims(image, axis=0),
                         prior_ent_model,
                         posterior_ent_model,
                         obs_ent_model,
                         prior_prec,
                         latent_prec,
                         obs_prec)

compressed_message = flatten_state(state)
compressed_length = 32 * (len(compressed_message) - len(init_bits))
print("Total compressed message length of {} bits at {} bits per pixel".format(
    compressed_length, compressed_length / (num_imgs * 784.0)))

retrieved_images = []
for i in range(num_imgs):
    state, obs = bbans_decode(state,
                              prior_ent_model,
                              posterior_ent_model,
                              obs_ent_model,
                              prior_prec,
                              latent_prec,
                              obs_prec)
    assert np.array_equal(test_data[num_imgs-i-1], np.array(obs))
assert np.array_equal(flatten_state(state), init_bits)
