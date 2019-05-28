from sklearn.decomposition import PCA


class PcaTransformer(object):
    def __init__(self, random_state):
        self.random_state = random_state
        self._transformer = None

    def fit(self, X, y):
        pca_obj = PCA(n_components=50, random_state=self.random_state)
        pca_obj.fit(X)

        self._transformer = pca_obj

        return self._transformer

    def transform(self, *args, **kwargs):
        return self._transformer.transform(*args, **kwargs)

