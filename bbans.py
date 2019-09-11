import rans.rans as r


def encode(state, symbol, entropy_model, precision, *model_inputs):
    starts, freqs = entropy_model.forward(symbol, *model_inputs)
    for start, freq in reversed(list(zip(starts, freqs))):
        r.encode(state, start, freq, precision)
    return state
