import os
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

from models.model import Model


class GaussianVAE(Model):
    def __init__(self,
                 obs_dim: int,
                 latent_dim: int,
                 encoder: Callable[[tf.Tensor, int], tf.Tensor],
                 decoder: Callable[[tf.Tensor], tf.Tensor],
                 decoder_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 optimiser: tf.train.Optimizer,
                 seed: int,
                 ) -> None:
        super().__init__(obs_dim, latent_dim)

        tf.set_random_seed(seed)

        obs = tf.placeholder(tf.float32, [None, obs_dim])
        latent_dist_params = encoder(obs, latent_dim)
        latent = self._build_sampled_latent(latent_dist_params)
        generated_dist_params = decoder(latent)
        _fixed_prior = tf.random_normal([1, latent_dim])

        loss = self._kl_divergence(latent_dist_params)
        loss += decoder_loss(obs, generated_dist_params)
        train_op = optimiser.minimize(loss)

        session = tf.Session()
        self._prior_fn = session.make_callable(_fixed_prior)
        self._posterior_fn = session.make_callable(
            latent_dist_params, [obs])
        self._generator_fn = session.make_callable(
            generated_dist_params, [obs])
        self._loss_fn = session.make_callable(loss, [obs])
        self._train_fn = session.make_callable(train_op, [obs])
        session.run(tf.global_variables_initializer())
        self.session = session
        self.saver = tf.train.Saver()

    def prior(self) -> np.ndarray:
        return self._prior_fn()

    def posterior(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist_params = self._posterior_fn(obs)
        mean, logstd = np.split(dist_params, 2, axis=-1)
        return mean, np.exp(logstd)

    def generate(self, latent: np.ndarray) -> np.ndarray:
        return self._generator_fn(latent)

    def elbo(self, data: np.ndarray) -> float:
        return self._loss_fn(data)

    def train(self, data: np.ndarray):
        return self._train_fn(data)

    def save_model(self, dir: str):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.saver.save(self.session, dir + 'model.checkpoint')

    def load_model(self, dir: str):
        chckpoint = tf.train.get_checkpoint_state(dir)
        if chckpoint is not None:
            self.saver.restore(self.session, chckpoint.model_checkpoint_path)
        else:
            print("No model checkpoint found in {}".format(dir))

    def _build_sampled_latent(self, enc_output):
        mean, logstd = tf.split(enc_output, num_or_size_splits=2, axis=-1)
        eps = tf.random_normal(tf.shape(mean))
        return mean + tf.exp(logstd) * eps

    def _kl_divergence(self, enc_output):
        mean, logstd = tf.split(enc_output, num_or_size_splits=2, axis=-1)
        kl_per_latent = -0.5 * (1 + 2 * logstd - (tf.exp(2 * logstd)
                                + tf.square(mean)))
        return tf.reduce_mean(tf.reduce_sum(kl_per_latent, axis=-1))


def diagonal_gaussian_encoder(x, latent_dim: int):
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(x, 100, activation=tf.nn.relu)
        return tf.layers.dense(x, 2 * latent_dim, activation=None)


def bernoulli_decoder(z):
    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(z, 100, activation=tf.nn.relu)
        return tf.layers.dense(x, 784, activation=None)


def sigmoid_cross_entropy_loss(targets, logits):
    cross_entropy_per_logit = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=logits)
    cross_entropy = tf.reduce_mean(
        tf.reduce_sum(cross_entropy_per_logit, axis=1))
    return cross_entropy
