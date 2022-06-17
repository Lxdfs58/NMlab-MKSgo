import itertools
def slice_deque(d, start, stop, step = 1):
    d.rotate(-start)
    slice = list(itertools.islice(d, 0, stop-start, step))
    d.rotate(start)
    return slice