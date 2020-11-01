import time
import timeit
import itertools
import random
from collections import deque
from sys import getsizeof, stderr

import numpy as np


def total_size(o, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint an object and all of its contents.
    From: https://code.activestate.com/recipes/577504/

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: itertools.chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def nesteddict_set(d, path, value):
    i, *path = path
    if not path:
        d[i] = value
        return
    if i not in d:
        d[i] = {}
    nesteddict_set(d[i], path, value)


def nesteddict_get(d, path):
    i, *path = path
    if not path:
        return d
    nesteddict_get(d[i], path)


class TimeitLoop:
    def __init__(self, repeat, measure=None):
        self.repeat = repeat
        self.measure = measure
        if self.measure is not None and not self.measure:
            self.measure.append(None)
        self.t0 = time.time()

    def __enter__(self):
        return range(self.repeat)

    def __exit__(self, type, value, traceback):
        self.t1 = time.time()
        if self.measure is not None:
            self.measure[0] = (self.t1 - self.t0) / self.repeat
        else:
            print((self.t1 - self.t0) / self.repeat)


class NestedDict:
    def __init__(self, size, group_size=2):
        self.d = {}
        for group in itertools.combinations(range(size), r=group_size):
            self[group] = random.random()

    def __getitem__(self, item):
        return nesteddict_get(self.d, item)

    def __setitem__(self, key, value):
        for perm in itertools.permutations(key):
            nesteddict_set(self.d, perm, value)


class NestedDictSorted:
    def __init__(self, size, group_size=2):
        self.d = {}
        for group in itertools.combinations(range(size), r=group_size):
            self[group] = random.random()

    def __getitem__(self, item):
        item = sorted(item)
        return nesteddict_get(self.d, item)

    def __setitem__(self, key, value):
        key = sorted(key)
        nesteddict_set(self.d, key, random.random())


class FrozenSetDict:
    def __init__(self, size, group_size=2):
        self.d = {}
        for group in itertools.combinations(range(size), r=group_size):
            self[group] = random.random()

    def __getitem__(self, item):
        return self.d[frozenset(item)]

    def __setitem__(self, key, value):
        self.d[frozenset(key)] = value


class TupleDict:
    def __init__(self, size, group_size=2):
        self.d = {}
        for group in itertools.combinations(range(size), r=group_size):
            self[group] = random.random()

    def __getitem__(self, item):
        return self.d[tuple(item)]

    def __setitem__(self, key, value):
        for perm in itertools.permutations(key):
            self.d[tuple(perm)] = value


class TupleDictSorted:
    def __init__(self, size, group_size=2):
        self.d = {}
        for group in itertools.combinations(range(size), r=group_size):
            value = random.random()
            group = sorted(group)
            self[group] = value

    def __getitem__(self, item):
        item = sorted(item)
        return self.d[tuple(item)]

    def __setitem__(self, key, value):
        item = sorted(key)
        self.d[tuple(item)] = value


def test_memory_usage(size, group):
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        struct = Struct(size, group)
        print(f"{total_size(struct.d):20d} {Struct.__name__}")


def test_init_time(size, group, repeat=10):
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        measure = []
        with TimeitLoop(repeat, measure) as loop:
            for x in loop:
                _ = Struct(size, group)
        print(f"{measure[0]:20f} {Struct.__name__}")


def test_write_all(size, group, repeat=10):
    indices = list(itertools.combinations(range(size), r=group))
    random.shuffle(indices)
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        struct = Struct(size, group)
        measure = []
        with TimeitLoop(repeat, measure) as loop:
            for x in loop:
                for i in indices:
                    struct[i] = 0
        print(f"{measure[0]:20f} {Struct.__name__}")


def test_access_all(size, group, repeat=10):
    indices = list(itertools.combinations(range(size), r=group))
    random.shuffle(indices)
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        struct = Struct(size, group)
        measure = []
        with TimeitLoop(repeat, measure) as loop:
            for x in loop:
                for i in indices:
                    _ = struct[i]
        print(f"{measure[0]:20f} {Struct.__name__}")


if __name__ == "__main__":
    size = (4+3+2+1)*10*10
    group = 2
    repeat = 100
    # test_init_time(size, group, repeat)
    # test_memory_usage(size, group)
    test_write_all(size, group, repeat)
    # test_access_all(size, group, repeat)
