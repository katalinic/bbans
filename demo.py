import argparse
import os

import numpy as np
import tensorflow as tf

from bbans import bbans_encode, bbans_decode
from entropy_model import DiagonalGaussianEntropyModel, BernoulliEntropyModel
import mnist_utils
from models.vae import (GaussianVAE, diagonal_gaussian_encoder,
                        bernoulli_decoder, sigmoid_cross_entropy_loss)
from quantiser import MaxEntropyGaussianQuantiser
from rans.rans import init_state, flatten_state, unflatten_state

parser = argparse.ArgumentParser(description='Configuration.')
parser.add_argument('-s', '--seed', default=42, type=int,
                    help='RNG seed.')
parser.add_argument('-n', '--num_imgs', default=100, type=int,
                    help='Number of images to encode and decode.')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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


def run(args):
    rng = np.random.RandomState(args.seed)
    init_bits = rng.randint(
        low=1 << 16, high=1 << 31, size=20, dtype=np.uint32)
    state = init_state()
    state = unflatten_state(init_bits)

    test_inds = np.arange(len(test_data))
    rng.shuffle(test_inds)
    inds_subset = test_inds[:args.num_imgs]

    for i, image in enumerate(test_data[inds_subset]):
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
    print("Compressed message length is {:.3f} bits per pixel.".format(
        compressed_length / (args.num_imgs * 784.0)))

    for i in range(args.num_imgs):
        state, obs = bbans_decode(state,
                                  prior_ent_model,
                                  posterior_ent_model,
                                  obs_ent_model,
                                  prior_prec,
                                  latent_prec,
                                  obs_prec)
        expected_img = test_data[inds_subset][args.num_imgs-i-1]
        assert np.array_equal(expected_img, np.array(obs))
    assert np.array_equal(flatten_state(state), init_bits)
    print("Reconstructed all compressed examples, as well as initial bits.")


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
