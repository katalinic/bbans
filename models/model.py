import abc

import numpy as np


class Model:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, obs_dim: int, latent_dim: int):
        self._obs_dim = obs_dim
        self._latent_dim = latent_dim

    @abc.abstractmethod
    def prior(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def posterior(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def generate(self, latent: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def elbo(self, data: np.ndarray) -> float:
        pass

    @abc.abstractmethod
    def train(self, data: np.ndarray):
        pass

    @abc.abstractmethod
    def save_model(self, dir: str):
        pass

    @abc.abstractmethod
    def load_model(self, dir: str):
        pass
