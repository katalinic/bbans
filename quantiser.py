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

    @abc.abstractmethod
    def quantise_backward(self, dist, cf, target_precision):
        pass

    @abc.abstractmethod
    def to_discrete(self, continuous_symbol):
        pass

    @abc.abstractmethod
    def to_continuous(self, discrete_symbol):
        pass


class MaxEntropyGaussianQuantiser(Quantiser):
    def __init__(self, precision):
        super().__init__(precision)
        self._bucket_endpoints = np.float32(norm.ppf(
            np.arange((1 << precision) + 1) / (1 << precision)))
        self._bucket_centres = np.float32(norm.ppf(
            (np.arange(1 << precision) + 0.5) / (1 << precision)))

    def quantise(self, dist, target_precision):
        return np.around(
            dist.cdf(self._bucket_endpoints) * (1 << target_precision)
            ).astype(int)

    def quantise_backward(self, dist, cf, target_precision):
        cp = (cf + 0.5) / (1 << target_precision)
        x = dist.ppf(cp)
        idx = np.searchsorted(self._bucket_endpoints, x, 'right') - 1
        return idx, np.around(dist.cdf(
            self._bucket_endpoints[idx: idx + 2]) * (1 << target_precision)
            ).astype(int)

    def to_discrete(self, continuous_symbol):
        return np.searchsorted(
            self._bucket_endpoints, continuous_symbol, 'right') - 1

    def to_continuous(self, discrete_symbol):
        return self._bucket_centres[discrete_symbol]
