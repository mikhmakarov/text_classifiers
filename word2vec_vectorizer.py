import numpy as np
import gensim

from gensim.models import Word2Vec


class Word2VecPretrainedVectorizer(object):
    def __init__(self):
        try:
            self.model = gensim.models.KeyedVectors.load('fasttext/model.model')
        except:
            self.model = None

    def fit(self, *args, **kwargs):
        return self

    def transform(self, docs):
        unknown = set()
        all_words = set()
        result = []
        for doc in docs:
            embds = []
            for word in doc:
                all_words.add(word)
                if word in self.model:
                    embds.append(np.array(self.model.get_vector(word)))
                else:
                    unknown.add(word)

            embds = np.array(embds)
            embds = np.mean(embds, axis=0)
            result.append(embds)

        matrix = np.concatenate(result).reshape(len(docs), 300)
        return matrix


class Word2VecCustomVectorizer(object):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        size = 300
        window_size = 3
        epochs = 100
        min_count = 2
        workers = 4

        # train word2vec model using gensim
        self.model = Word2Vec(X, sg=1, window=window_size, size=size, min_count=min_count, workers=workers, iter=epochs)

        return self

    def transform(self, docs):
        unknown = set()
        all_words = set()
        result = []
        for doc in docs:
            embds = []
            for word in doc:
                all_words.add(word)
                if word in self.model:
                    embds.append(np.array(self.model.wv.get_vector(word)))
                else:
                    unknown.add(word)

            embds = np.array(embds)
            embds = np.mean(embds, axis=0)
            result.append(embds)

        matrix = np.concatenate(result).reshape(len(docs), 300)
        return matrix
