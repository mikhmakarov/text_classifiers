import numpy as np

from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder

elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz")


class ElmoPretrainedVectorizer(object):
    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        return self

    def transform(self, docs):
        return np.array(elmo(docs.tolist()))

