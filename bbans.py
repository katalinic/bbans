import numpy as np

import rans.rans as r


def encode(state, symbol, entropy_model, precision, *model_inputs):
    starts, freqs = entropy_model.forward(symbol, *model_inputs)
    for start, freq in reversed(list(zip(starts, freqs))):
        state = r.encode(state, start, freq, precision)
    return state


def decode(state, entropy_model, precision, *model_inputs):
    symbols = []
    done = False
    while not done:
        cf = r.peek(state, precision)
        start, freq, symbol, done = entropy_model.backward(cf, *model_inputs)
        assert start <= cf < start + freq
        _, state = r.decode(state, start, freq, precision)
        symbols.append(symbol)
    return state, np.asarray(symbols)


def bbans_encode(state, observation, prior, posterior, likelihood,
                 prior_prec, posterior_prec, likelihood_prec):
    state, latent = decode(state, posterior, posterior_prec, observation)
    state = encode(state, observation, likelihood, likelihood_prec, latent)
    state = encode(state, latent, prior, prior_prec)
    return state


def bbans_decode(state, prior, posterior, likelihood,
                 prior_prec, posterior_prec, likelihood_prec):
    state, latent = decode(state, prior, prior_prec)
    state, observation = decode(state, likelihood, likelihood_prec, latent)
    state = encode(state, latent, posterior, posterior_prec, observation)
    return state, observation
