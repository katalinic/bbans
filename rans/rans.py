from collections import deque

BLOCK_SIZE = 32
RANS_L = 1 << (BLOCK_SIZE - 1)
TAIL_BITS = (1 << BLOCK_SIZE) - 1


def init_state():
    return RANS_L, deque()


def encode(x: tuple, start: int, freq: int, precision: int):
    assert freq != 0
    x_max = ((RANS_L >> precision) << BLOCK_SIZE) * freq
    if x[0] >= x_max:
        x[1].appendleft(x[0] & TAIL_BITS)
        x = x[0] >> BLOCK_SIZE, x[1]
        assert x[0] < x_max
    return ((x[0] // freq) << precision) + (x[0] % freq) + start, x[1]


def decode(x: tuple, start: int, freq: int, precision: int):
    mask = (1 << precision) - 1
    ccf = x[0] & mask
    x = freq * (x[0] >> precision) + (x[0] & mask) - start, x[1]
    if x[0] < RANS_L:
        assert len(x[1]) > 0
        block = x[1].popleft()
        x = (x[0] << BLOCK_SIZE) | block, x[1]
        assert x[0] >= RANS_L

    return ccf, x


def flatten_state(x: tuple):
    return [x[0] >> BLOCK_SIZE, x[0]] + list(x[1])


def unflatten_state(x: list):
    return (x[0] << BLOCK_SIZE) | x[1], deque(x[2:])
