
from typing import Tuple


def index_to_multidim(idx, dims):
    indices = []
    off = idx
    stride = 1
    dims = dims[1:]
    for dim in dims[::-1]:
        next_stride = dim * stride
        mod = off % next_stride
        off -= mod
        indices.insert(0, mod // stride)
        stride = next_stride
    indices.insert(0, off // stride)
    return indices


def xgcd(a: int, b: int) -> Tuple[int, int, int]:
    """return (g, x, y) such that a*x + b*y = g = gcd(a, b)"""
    x0, x1, y0, y1 = 0, 1, 1, 0
    while a != 0:
        (q, a), b = divmod(b, a), a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    return b, x0, y0


def modinv(a: int, b: int) -> int:
    """return x such that (x * a) % b == 1"""
    g, x, _ = xgcd(a, b)
    if g != 1:
        raise Exception('gcd(a, b) != 1')
    return x % b
