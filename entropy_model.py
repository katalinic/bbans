import abc
import numbers

import numpy as np
from scipy.stats import norm, bernoulli


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

        symbol = self.quantiser.to_discrete(symbol)

        mean, std = self.distribution_fn(*model_inputs)
        mean = np.ravel(mean)
        std = np.ravel(std)

        cdfs = [self.quantiser.quantise(
            norm(m, s), self.precision) for m, s in zip(mean, std)]
        cdfs = np.asarray(cdfs)

        num_symbols = symbol.size
        starts = cdfs[np.arange(num_symbols), symbol]
        freqs = cdfs[np.arange(num_symbols), symbol + 1] - starts
        starts = np.ravel(starts)
        freqs = np.ravel(freqs)

        return starts.tolist(), freqs.tolist()

    def backward(self, cf, *model_inputs):
        assert isinstance(cf, numbers.Integral), type(cf)

        if self._cached_params is not None:
            mean, std = self._cached_params
        else:
            mean, std = self.distribution_fn(*model_inputs)
            mean = np.ravel(mean)
            std = np.ravel(std)
            self._cached_params = (mean, std)
            self._decoding_idx = 0

        m, s = mean[self._decoding_idx], std[self._decoding_idx]
        symbol, cdf = self.quantiser.quantise_backward(
            norm(m, s), cf, self.precision)

        start = cdf[0]
        freq = cdf[1] - start

        self._decoding_idx += 1
        if self._decoding_idx == len(mean):
            self._cached_params = None

        symbol = self.quantiser.to_continuous(symbol)
        return start, freq, symbol, bool(self._decoding_idx == len(mean))


class BernoulliEntropyModel(EntropyModel):
    def __init__(self, distribution_fn, quantiser=None, precision=12):
        super().__init__(distribution_fn, quantiser, precision)
        self._cached_params = None
        self._decoding_idx = -1

    def forward(self, symbol, *model_inputs):
        assert np.issubdtype(symbol.dtype, numbers.Integral)
        symbol = symbol.ravel()

        p1 = self.distribution_fn(*model_inputs).ravel()

        starts, freqs = [], []
        for p, s in zip(p1, symbol):
            cdf = self._cdf(p)
            starts.append(cdf[s])
            freqs.append(cdf[s + 1] - cdf[s])

        return starts, freqs

    def backward(self, cf, *model_inputs):
        assert isinstance(cf, numbers.Integral), type(cf)

        if self._cached_params is not None:
            p1 = self._cached_params
        else:
            p1 = self.distribution_fn(*model_inputs)
            p1 = np.ravel(p1)
            self._cached_params = p1
            self._decoding_idx = 0

        _p1 = p1[self._decoding_idx]
        cdf = self._cdf(_p1)
        symbol = np.searchsorted(cdf, cf, 'right') - 1

        start = cdf[symbol]
        freq = cdf[symbol + 1] - start

        self._decoding_idx += 1
        if self._decoding_idx == len(p1):
            self._cached_params = None

        return start, freq, symbol, bool(self._decoding_idx == len(p1))

    def _cdf(self, p1):
        cdf = bernoulli.cdf([0, 1], p1) * (1 << self.precision)
        cdf = np.insert(np.around(cdf).astype(int), 0, 0)
        # Prevent overflow.
        cdf[1] = np.clip(cdf[1], 1, (1 << self.precision) - 1)
        return cdf
