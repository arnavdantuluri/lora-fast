#Taken from stable-fast: https://github.com/chengzeyi/stable-fast/tree/main

import triton
import triton.language as tl
import copy
import types
import functools

@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    # w2_over_w = weight_2 / new_weight
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


def copy_func(f, globals=None, module=None, name=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    if name is None:
        name = f.__name__
    g = types.FunctionType(f.__code__,
                           globals,
                           name=name,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    if module is not None:
        g.__module__ = module
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    g.__name__ = name
    return g