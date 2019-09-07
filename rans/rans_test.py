import numpy as np

import rans


def test():
    rng = np.random.RandomState(0)

    K = 8
    L = 1000
    starts = rng.randint(0, 2**K, size=L)
    freqs = rng.randint(1, 2**K, size=L) % (2**K - starts)
    freqs[freqs == 0] = 1
    assert np.all(starts + freqs <= 2**K)
    print("Exact entropy: {:.2f} bits.".format(np.sum(np.log2(2**K / freqs))))

    state = rans.init_state()
    for start, freq in zip(starts, freqs):
        state = rans.encode(state, start, freq, K)
    coded_arr = rans.flatten_state(state)
    print("Actual output size: " + str(32 * len(coded_arr)) + " bits.")

    state = rans.unflatten_state(coded_arr)
    for start, freq in reversed(list(zip(starts, freqs))):
        cf, state = rans.decode(state, start, freq, K)
        assert start <= cf < start + freq
    assert state[0] == rans.RANS_L
    assert not state[1]


# test()
