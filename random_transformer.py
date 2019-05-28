import numpy as np


class RandomTransformer(object):
    def fit(self, X, y):
        return self

    def transform(self, *args, **kwargs):
        return np.random.randn(*args[0].shape, 300)
