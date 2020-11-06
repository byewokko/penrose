import time
import timeit
import itertools
import random
from collections import deque
from sys import getsizeof, stderr

import numpy as np


"""
Benchmarking of different n-dimensional dict implementations.

Guess I'm not using any of them after all.
"""


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
    def __init__(self, data=None):
        self._d = {}
        if data:
            if isinstance(data, dict):
                # TODO: verify
                self._d = data

    def __getitem__(self, item):
        try:
            return nesteddict_get(self._d, item)
        except KeyError:
            return None

    def __setitem__(self, key, value):
        for perm in itertools.permutations(key):
            nesteddict_set(self._d, perm, value)

    def extract(self, key):
        return NestedDict(self[(key,)])


class NestedDictSorted:
    def __init__(self, data=None):
        self._d = {}
        if data:
            if isinstance(data, dict):
                # TODO: verify
                self._d = data

    def __getitem__(self, item):
        item = sorted(item)
        try:
            return nesteddict_get(self._d, item)
        except KeyError:
            return None

    def __setitem__(self, key, value):
        key = sorted(key)
        nesteddict_set(self._d, key, random.random())

    def extract(self, key):
        def extract(d, key):
            # if not isinstance(d, dict):
            #     return
            d_new = {}
            for k in d.keys():
                if k == key:
                    if isinstance(d[k], dict):
                        d_new.update(d[k])
                    else:
                        return d[k]
                elif k < key:
                    if isinstance(d[k], dict):
                        tmp = extract(d[k], key)
                        if tmp:
                            d_new[k] = tmp
            # print(d_new)
            return d_new
        return NestedDictSorted(extract(self._d, key))


class FrozenSetDict:
    def __init__(self, data=None):
        self._d = {}
        if data:
            if isinstance(data, dict):
                # TODO: verify
                self._d = data

    def __getitem__(self, item):
        try:
            return self._d[frozenset(item)]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        self._d[frozenset(key)] = value

    def extract(self, key):
        d_new = {}
        difset = {key}
        for k, v in self._d.items():
            if key in k:
                d_new[k.difference(difset)] = v
        return FrozenSetDict(d_new)

    def values(self):
        return self._d.values()

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()


class TupleDict:
    def __init__(self, data=None):
        self._d = {}
        if data:
            if isinstance(data, dict):
                # TODO: verify
                self._d = data

    def __getitem__(self, item):
        try:
            return self._d[tuple(item)]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        for perm in itertools.permutations(key):
            self._d[tuple(perm)] = value

    def extract(self, key):
        d_new = {}
        for k, v in self._d.items():
            if key in k:
                d_new[tuple(i for i in k if i != key)] = v
        return TupleDict(d_new)


class TupleDictSorted:
    def __init__(self, data=None):
        self._d = {}
        if data:
            if isinstance(data, dict):
                # TODO: verify
                self._d = data

    def __getitem__(self, item):
        item = sorted(item)
        try:
            return self._d[tuple(item)]
        except KeyError:
            return None

    def __setitem__(self, key, value):
        item = sorted(key)
        self._d[tuple(item)] = value

    def extract(self, key):
        d_new = {}
        for k, v in self._d.items():
            if key in k:
                d_new[tuple(i for i in k if i != key)] = v
        return TupleDictSorted(d_new)


def test_memory_usage(size, group):
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        struct = Struct()
        print(f"{total_size(struct._d):20d} {Struct.__name__}")


def test_extract_time(size, group, repeat=10):
    indices = list(itertools.combinations(range(size), r=group))
    values = [random.random() for _ in range(len(indices))]
    random.shuffle(indices)
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        measure = []
        struct = Struct()
        for i, v in zip(indices, values):
            struct[i] = v
        with TimeitLoop(repeat, measure) as loop:
            for _ in loop:
                for k in range(size):
                    ex = struct.extract(k)
        print(f"{measure[0]:20f} {Struct.__name__}")


def test_write_all(size, group, repeat=10):
    indices = list(itertools.combinations(range(size), r=group))
    values = [random.random() for _ in range(len(indices))]
    random.shuffle(indices)
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        struct = Struct()
        measure = []
        with TimeitLoop(repeat, measure) as loop:
            for x in loop:
                for i, v in zip(indices, values):
                    struct[i] = v
        print(f"{measure[0]:20f} {Struct.__name__}")


def test_access_all(size, group, repeat=10):
    indices = list(itertools.combinations(range(size), r=group))
    values = [random.random() for _ in range(len(indices))]
    random.shuffle(indices)
    for Struct in (NestedDict, NestedDictSorted, TupleDict, TupleDictSorted, FrozenSetDict):
        struct = Struct()
        for i, v in zip(indices, values):
            struct[i] = v
        measure = []
        with TimeitLoop(repeat, measure) as loop:
            for x in loop:
                for i in indices:
                    _ = struct[i]
        print(f"{measure[0]:20f} {Struct.__name__}")


def main():
    size = (4 + 3 + 2 + 1) * 10 #* 10
    group = 3
    repeat = 5
    test_extract_time(size, group, repeat)
    # test_memory_usage(size, group)
    # test_write_all(size, group, repeat)
    # test_access_all(size, group, repeat)


if __name__ == "__main__":
    main()
