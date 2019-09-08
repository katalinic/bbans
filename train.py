import argparse
import os

import tensorflow as tf

import mnist_utils
from models.vae import (GaussianVAE, diagonal_gaussian_encoder,
                        bernoulli_decoder, sigmoid_cross_entropy_loss)

parser = argparse.ArgumentParser(description='Configuration.')
parser.add_argument('-t', '--training', default=False, action="store_true",
                    help='Set to True for training.')
parser.add_argument('-e', '--epochs', default=20, type=int,
                    help='Number of training epochs.')
parser.add_argument('-b', '--batch_size', default=100, type=int,
                    help='Number of positions to click.')
parser.add_argument('-d', '--dir', default='./saved_models', type=str,
                    help='Directory of saved models.')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train(args):
    model = GaussianVAE(
        obs_dim=784,
        latent_dim=40,
        encoder=diagonal_gaussian_encoder,
        decoder=bernoulli_decoder,
        decoder_loss=sigmoid_cross_entropy_loss,
        optimiser=tf.train.AdamOptimizer(1e-3),
        seed=1)

    data = mnist_utils.load_train()
    data = mnist_utils.binarise(data)

    test_data = mnist_utils.load_test()
    test_data = mnist_utils.binarise(test_data)

    for epoch in range(1, args.epochs + 1):
        for batch in mnist_utils.generate_batches(data, args.batch_size, 0):
            assert batch.shape == (args.batch_size, 784)
            model.train(batch)
        print("Completed epoch {} of {}.".format(epoch, args.epochs))
        elbo = model.elbo(test_data)
        print("ELBO: {:2f} - ELBO / dim {:2f}".format(elbo, elbo / 784))
        model.save_model(args.dir)


def test(args):
    model = GaussianVAE(
        obs_dim=784,
        latent_dim=40,
        encoder=diagonal_gaussian_encoder,
        decoder=bernoulli_decoder,
        decoder_loss=sigmoid_cross_entropy_loss,
        optimiser=tf.train.AdamOptimizer(1e-3),
        seed=0)
    model.load_model(args.dir)

    test_data = mnist_utils.load_test()
    test_data = mnist_utils.binarise(test_data)

    elbo = model.elbo(test_data)
    print("ELBO: {:2f} - ELBO / dim {:2f}".format(elbo, elbo / 784))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.training:
        print("Training.")
        train(args)
    else:
        print("Evaluation.")
        test(args)
