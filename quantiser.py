import abc

import numpy as np
from scipy.stats import norm


class Quantiser:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, precision):
        self.precision = precision

    @abc.abstractmethod
    def quantise(self, dist, target_precision):
        pass


class MaxEntropyGaussianQuantiser(Quantiser):
    def __init__(self, precision):
        super().__init__(precision)
        self._bucket_endpoints = np.float32(norm.ppf(
            np.arange((1 << precision) + 1) / (1 << precision)))
        self._bucket_centres = {}

    def quantise(self, dist, target_precision):
        return np.around(
            dist.cdf(self._bucket_endpoints) * (1 << target_precision)
            ).astype(int)
