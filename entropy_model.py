import abc
import numbers

import numpy as np
from scipy.stats import norm


class EntropyModel:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, distribution_fn, quantiser=None, precision=12):
        self.distribution_fn = distribution_fn
        self.quantiser = quantiser
        self.precision = precision

    @abc.abstractmethod
    def forward(self, *model_inputs):
        """Returns starts, freqs."""
        pass

    @abc.abstractmethod
    def backward(self, cf, *model_inputs):
        """Returns start, freq, symbol, done."""
        pass


class DiagonalGaussianEntropyModel(EntropyModel):
    def __init__(self, distribution_fn, quantiser=None, precision=12):
        if quantiser is None:
            print("Missing quantiser for continuous distribution.")
            return
        super().__init__(distribution_fn, quantiser, precision)
        self._cached_params = None
        self._decoding_idx = -1

    def forward(self, symbol, *model_inputs):
        assert np.issubdtype(symbol.dtype, numbers.Integral)
        assert symbol.ndim == 1
        assert len(model_inputs) == 1

        mean, std = self.distribution_fn(model_inputs[0])
        mean = np.ravel(mean)
        std = np.ravel(std)

        quantised_dists = [self.quantiser.quantise(
            norm(m, s), self.precision) for m, s in zip(mean, std)]
        quantised_dists = np.array(quantised_dists)
        assert quantised_dists.ndim == 2

        num_symbols = symbol.size
        starts = quantised_dists[np.arange(num_symbols), symbol]
        freqs = quantised_dists[np.arange(num_symbols), symbol + 1] - starts

        starts = np.ravel(starts)
        freqs = np.ravel(freqs)

        starts[starts == 1 << self.precision] = (1 << self.precision) - 1
        freqs[freqs == 0] = 1

        return starts.tolist(), freqs.tolist()

    def backward(self, cf, *model_inputs):
        assert isinstance(cf, numbers.Integral), type(cf)
        assert len(model_inputs) == 1

        if self._cached_params is not None:
            mean, std = self._cached_params
        else:
            mean, std = self.distribution_fn(model_inputs[0])
            mean = np.ravel(mean)
            std = np.ravel(std)
            self._cached_params = (mean, std)
            self._decoding_idx = len(mean) - 1

        m, s = mean[self._decoding_idx], std[self._decoding_idx]
        symbol, starts = self.quantiser.quantise_backward(
            norm(m, s), cf, self.precision)
        assert starts.ndim == 1
        assert starts.size == 2

        freq = max(1, int(starts[1] - starts[0]))
        start = min((1 << self.precision) - 1, int(starts[0]))
        self._decoding_idx -= 1
        
        return start, freq, int(symbol), bool(self._decoding_idx < 0)
