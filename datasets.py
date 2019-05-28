import os

from collections import defaultdict

import pymorphy2
import razdel

from russian_tagsets import converters

from stop_words import get_stop_words

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from word2vec_vectorizer import Word2VecPretrainedVectorizer, Word2VecCustomVectorizer
from elmo_vectorizer import ElmoPretrainedVectorizer
from pca_transformer import PcaTransformer
from random_transformer import RandomTransformer

from tqdm import tqdm


DATA_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/news')
SEED = 10


class Dataset(object):
    def __init__(self,
                 doc_type='Lenta',
                 use_title=False,
                 rubrics=None,
                 subsample=1.0,
                 normalize_words=True,
                 rm_stop_words=True,
                 random_state=SEED,
                 min_len_doc_threshold=3,
                 max_len_doc_threshold=1000):
        self.morph = pymorphy2.MorphAnalyzer(result_type=None)
        self.label_encoder = LabelEncoder()

        self.count_vectorizer = CountVectorizer(tokenizer=self._identity, preprocessor=self._identity, min_df=10)
        self.tfidf_vectorized = TfidfVectorizer(tokenizer=self._identity, preprocessor=self._identity, min_df=10)
        self.word2vec_vectorizer = Word2VecPretrainedVectorizer()
        self.word2vec_custom_vectorizer = Word2VecCustomVectorizer()
        self.elmo_vectorizer = ElmoPretrainedVectorizer()
        self.pca_transformer = PcaTransformer(random_state=random_state)
        self.random_transformer = RandomTransformer()

        self.standard_scaler = StandardScaler()
        self.reduce_dimension = None
        self.standardize = None

        self.root_path = os.path.join(DATA_FOLDER, doc_type)
        self.transform_type = None

        self.doc_type = doc_type
        self.use_title = use_title
        self.rubrics = rubrics
        self.subsample = subsample
        self.rm_stop_words = rm_stop_words
        self.random_state = random_state
        self.normalize_words = normalize_words
        self.min_len_doc_threshold = min_len_doc_threshold
        self.max_len_doc_threshold = max_len_doc_threshold

        self.metadata = self._load_metadata()
        self.docs = self._load_docs()
        self.docs['text'] = self._process_docs(self.docs['text'].tolist())
        self.docs = self.docs[self.docs['text'].str.len() >= self.min_len_doc_threshold]
        self.docs = self.docs[self.docs['text'].str.len() <= self.max_len_doc_threshold]

        self.label_encoder.fit(self.docs['label'])

        self.X = None
        self.y = self.label_encoder.transform(self.docs['label'])

    def get_classes_balance(self, threshold=1000):
        cnt = self.metadata.groupby('textrubric').size()
        cnt = cnt[cnt >= threshold]

        return cnt

    def get_numb_words_dist(self):
        return self._get_numb_words_dist()

    def get_numb_words_class_dist(self, rubrics):
        if not isinstance(rubrics, list):
            rubrics = [rubrics]

        return self._get_numb_words_dist(rubrics)

    def get_word_freq_dist(self):
        freq = defaultdict(int)

        for doc in self.docs['text']:
            for word in doc:
                freq[word] += 1

        return freq.values()

    def get_top_words(self, n=10):
        return self._get_top_words(None, n)

    def get_top_ngrams(self, n_words, ngram):
        return self._get_top_words(rubrics=None, n=n_words, ngram=ngram)

    def get_top_words_class(self, rubrics, n=10):
        if not isinstance(rubrics, list):
            rubrics = [rubrics]

        return self._get_top_words(rubrics, n)

    def get_pos_dist(self):
        pos = defaultdict(int)
        for doc in self.docs['text']:
            for word in doc:
                p = self.morph.parse(word)[0][1].POS
                if p is not None:
                    pos[self.morph.lat2cyr(p)] += 1

        return pos

    def get_data(self):
        self.X = self.docs['text']
        self.y = self.docs['label']

        return self.X, self.y

    def get_class_labels(self):
        return self.label_encoder.classes_

    def get_transform_pipeline(self, clf, transform_type, standardize=True, pca=False):
        steps = [('extract_features', self._get_transformer(transform_type))]

        if transform_type in ['bow', 'tf-idf']:
            steps.append(('to_dense', FunctionTransformer(self._to_dense, accept_sparse=True)))

        if standardize or pca:
            steps.append(('standardization', self.standard_scaler))

            if pca:
                steps.append(('pca', self.pca_transformer))

        steps.append(('classifier', clf))
        pipeline = Pipeline(steps)

        return pipeline

    @staticmethod
    def _to_dense(x):
        return x.todense()

    @staticmethod
    def _identity(x):
        return x

    def _get_top_words(self, rubrics=None, n=10, ngram=1):
        if rubrics is None:
            rubrics = self.rubrics

        freq = defaultdict(int)

        docs = self.docs[self.docs['label'].isin(rubrics)]['text']
        docs = [self._get_ngrams(doc, ngram) for doc in docs]

        for doc in docs:
            for word in doc:
                freq[word] += 1

        top_words = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:n]

        return {w: freq[w] for w in top_words}

    def _get_numb_words_dist(self, rubrics=None):
        if rubrics is None:
            rubrics = self.rubrics

        numb_words = []
        for doc in self.docs[self.docs['label'].isin(rubrics)]['text']:
            numb_words.append(len(doc))

        return numb_words

    def _get_ngrams(self, input_list, n):
        return [' '.join(t) for t in zip(*[input_list[i:] for i in range(n)])]

    def _process_docs(self, docs):
        stop_words = get_stop_words('ru')
        res = []
        for doc in tqdm(docs, desc='processing docs'):
            words = [t.text.lower() for t in razdel.tokenize(doc)]
            words = [w for w in words if len(w) > 3]
            words = [w for w in words if w.isalpha()]

            if self.rm_stop_words:
                words = [w for w in words if w not in stop_words]

            if self.normalize_words:
                words = [self.morph.parse(w)[0][2] for w in words]

            res.append(words)

        return res

    def _load_metadata(self):
        metadata_path = os.path.join(self.root_path, 'newmetadata.csv')
        metadata = pd.read_csv(metadata_path, sep='\t')

        if self.rubrics is not None:
            metadata = metadata[metadata['textrubric'].isin(self.rubrics)].copy().reset_index(drop=True)

        return metadata

    def _load_docs(self):
        if self.use_title:
            docs = self.metadata[['textname', 'textrubric']].copy()
            docs.columns = ['text', 'label']

            if self.subsample < 1:
                dfs = []
                for c in docs['label'].unique():
                    data = docs[docs['label'] == c].sample(frac=self.subsample, random_state=self.random_state)
                    dfs.append(data)

                docs = pd.concat(dfs).reset_index(drop=True)
        else:
            texts = []
            labels = []
            for rubric in tqdm(self.metadata['textrubric'].unique(), desc='loading rubrics'):
                rubric_docs = self.metadata[self.metadata['textrubric'] == rubric]

                if self.subsample < 1.0:
                    rubric_ids = rubric_docs.sample(frac=self.subsample, random_state=self.random_state)['textid']
                    rubric_ids = rubric_ids.tolist()
                else:
                    rubric_ids = rubric_docs['textid'].tolist()

                for doc_id in rubric_ids:
                    filename = os.path.join(self.root_path, 'texts', doc_id + '.txt')
                    try:
                        with open(filename) as doc_file:
                            texts.append(doc_file.read())
                            labels.append(rubric)
                    except FileNotFoundError:
                        print('file not found ' + filename)

            docs = pd.DataFrame({'text': texts, 'label': labels})

        return docs

    def _tokenizer(self, text):
        return [self.morph.parse(t.text.lower())[0][2] for t in razdel.tokenize(text)]

    def _get_transformer(self, transform_type):
        if transform_type == 'bow':
            return self.count_vectorizer
        elif transform_type == 'tf-idf':
            return self.tfidf_vectorized
        elif transform_type == 'word2vec':
            return self.word2vec_vectorizer
        elif transform_type == 'word2vec_custom':
            return self.word2vec_custom_vectorizer
        elif transform_type == 'elmo':
            return self.elmo_vectorizer
        elif transform_type == 'dummy':
            return self.random_transformer
        else:
            raise ValueError('incorrect transform type')


if __name__ == '__main__':
    rubrics = ['Мир', 'Россия', 'Политика']
    ds = Dataset(use_title=False, rubrics=rubrics, normalize_words=True, subsample=0.1)

    X, y = ds.get_data()
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    clf = OneVsRestClassifier(LogisticRegression())
    pipeline = ds.get_transform_pipeline(clf, 'elmo')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    pipeline.fit(X_train, y_train)
    y_predicted = pipeline.predict(X_test)
    print(accuracy_score(y_test, y_predicted))

    pipeline = ds.get_transform_pipeline(clf, 'bow', pca=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    pipeline.fit(X_train, y_train)
    y_predicted = pipeline.predict(X_test)
    print(accuracy_score(y_test, y_predicted))
    pass

