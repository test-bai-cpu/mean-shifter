class NotFittedError(Exception):
    # Exception class to raise if mean-shift is used before fitting.
    pass


class KernelInputError(ValueError):
    # Exception class to raise if given kernel info is not correct.
    pass
