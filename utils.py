import os
import sys
import time


class Timer:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def __enter__(self): self.start = time.time(); return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval, self.unit = self.__getInterval()
        if self.verbose: print(f"The query took {self.interval:.5f} {self.unit}.")

    def __getInterval(self) -> tuple:
        interval = self.end - self.start
        if interval == 0.0: return interval, 's'
        if 0.000_001 > interval: return interval * 1_000_000_000, 'ns'
        if 0.001 > interval >= 0.000_001: return interval * 1_000_000, 'μs'
        if 1.0 > interval >= 0.001: return interval * 1_000, 'ms'
        return interval, 'seconds'

    def getAbsoluteInterval(self) -> float: return self.end - self.start

    def print(self, name=None) -> None:
        if name: print(f"The query {name} took {self.interval:.5f} {self.unit}.")
        else: print(f"The query took {self.interval:.5f} {self.unit}.")


class BlockPrint:
    def __init__(self): pass

    def __enter__(self): blockPrint(); return self

    def __exit__(self, *args): enablePrint()


def blockPrint(): sys.stdout = open(os.devnull, 'w')


def enablePrint(): sys.stdout = sys.__stdout__
