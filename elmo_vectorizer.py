from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder


class ElmoPretrainedVectorizer(object):
    def __init__(self):
        self.elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-wiki_600k_steps.tar.gz")

    def fit(self, *args, **kwargs):
        return self

    def transform(self, docs):
        return self.elmo(docs.tolist())
