import rans.rans as r


def encode(state, symbol, entropy_model, precision, *model_inputs):
    starts, freqs = entropy_model.forward(symbol, *model_inputs)
    for start, freq in zip(starts, freqs):
        state = r.encode(state, start, freq, precision)
    return state


def decode(state, entropy_model, precision, *model_inputs):
    symbols = []
    done = False
    while not done:
        cf = r.peek(state, precision)
        start, freq, symbol, done = entropy_model.backward(cf, *model_inputs)
        _, state = r.decode(state, start, freq, precision)
        symbols.append(symbol)
    return state, symbols
