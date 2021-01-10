"""A torch.cuda.Event() based timer."""

from itertools import tee
import torch


class CUDATimer:
    def __init__(self):
        super().__init__()
        self.events = []

    def __len__(self):
        return len(self.events)

    def record(self, label=None):
        if label is None:
            label = str(self.__len__())
        new_event = torch.cuda.Event(enable_timing=True)
        new_event.record()
        self.events.append((new_event, label))

    def stop(self):
        elapsed_times = {}

        if self.__len__() >= 2:
            torch.cuda.synchronize()
            for (e_a, l), (e_b, _) in pairwise(self.events):
                elapsed_times[l] = e_a.elapsed_time(e_b)

        self.clear()

        return elapsed_times

    def clear(self):
        self.events = []


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
