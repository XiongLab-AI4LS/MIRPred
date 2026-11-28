def chunk_starts(length: int, maximum: int, overlap: int) -> list[int]:
    if length <= maximum:
        return [0]
    stride = maximum - overlap
    starts = list(range(0, length - maximum + 1, stride))
    final = length - maximum
    if starts[-1] != final:
        starts.append(final)
    return starts
