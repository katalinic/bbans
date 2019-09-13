"""Based on https://github.com/rygorous/ryg_rans/blob/master/rans64.h"""
from collections import deque, namedtuple
from typing import Sequence, Tuple

BLOCK_SIZE = 32
RANS_L = 1 << (BLOCK_SIZE - 1)
TAIL_BITS = (1 << BLOCK_SIZE) - 1


rANSstate = namedtuple('rANSstate', 'Buffer Stream')


def init_state(state: list = None):
    if state is None:
        return rANSstate(Buffer=RANS_L, Stream=deque())
    else:
        return rANSstate(Buffer=RANS_L, Stream=deque(state))


def encode(state: rANSstate,
           start: int,
           freq: int,
           precision: int) -> rANSstate:
    assert freq != 0
    buffer_ubound = ((RANS_L >> precision) << BLOCK_SIZE) * freq
    if state.Buffer >= buffer_ubound:
        state.Stream.appendleft(state.Buffer & TAIL_BITS)
        state = rANSstate(Buffer=state.Buffer >> BLOCK_SIZE,
                          Stream=state.Stream)
        assert state.Buffer < buffer_ubound
    return rANSstate(Buffer=((state.Buffer // freq) << precision) +
                            (state.Buffer % freq) + start,
                     Stream=state.Stream)


def decode(state: rANSstate,
           start: int,
           freq: int,
           precision: int) -> Tuple[int, rANSstate]:
    mask = (1 << precision) - 1
    ccf = state.Buffer & mask
    state = rANSstate(Buffer=freq * (state.Buffer >> precision) +
                      (state.Buffer & mask) - start,
                      Stream=state.Stream)
    if state.Buffer < RANS_L:
        assert state.Stream
        block = state.Stream.popleft()
        state = rANSstate(Buffer=(state.Buffer << BLOCK_SIZE) | block,
                          Stream=state.Stream)
        assert state.Buffer >= RANS_L
    return ccf, state


def peek(state: rANSstate, precision: int) -> int:
    return state.Buffer & ((1 << precision) - 1)


def flatten_state(state: rANSstate) -> Sequence[int]:
    return ([state.Buffer & TAIL_BITS, state.Buffer >> BLOCK_SIZE] +
            list(state.Stream))


def unflatten_state(arr: Sequence[int]) -> rANSstate:
    return rANSstate(Buffer=arr[0] | (arr[1] << BLOCK_SIZE),
                     Stream=deque(arr[2:]))
