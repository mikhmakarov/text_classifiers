import json

import imp
import datasets

import numpy as np
import os
import tensorflow as tf

imp.reload(datasets)

from datasets import Dataset

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score

# models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


SEED = 10


RES_FOLDER = 'results/'


def save_result(name, res):
    if not os.path.exists(RES_FOLDER):
        os.makedirs(RES_FOLDER)

    with open(RES_FOLDER + name + '.json', 'w') as output:
        json.dump(res, output)


def main():
    rubrics = ['Мир', 'Россия', 'Политика', 'Экономика', 'Наука и техника', 'Украина',
               'Госэкономика', 'Спорт', 'Общество', 'Бывший СССР', 'Культура', 'Медиа',
               'Футбол', 'Музыка', 'Наука']
    lenta = Dataset(use_title=False, rubrics=rubrics, random_state=SEED, subsample=0.1)

    X, y = lenta.get_data()

    models = [
        LogisticRegression(random_state=SEED),
        MultinomialNB(),
        GaussianNB(),
        SVC(),
        RandomForestClassifier(),
        XGBClassifier()
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    transformations = ['elmo']

    with tf.device("/gpu:0"):
        # Setup operations
        for trans in transformations:
            for model in models:
                clf = model

                pipeline = lenta.get_transform_pipeline(clf, trans, standardize=False)
                scores = cross_val_score(pipeline, X, y, verbose=10, cv=cv, n_jobs=1)
                mean = np.mean(scores)
                std = np.std(scores)
                res = {'mean': round(mean, 3), 'std': round(std, 3)}
                msg = 'Transformation: {}, Model: {}, accuracy {:.3f}(+- {:.3f})'
                print(msg.format(trans, model, mean, std))
                save_result(type(model).__name__ + '_' + trans, res)


if __name__ == '__main__':
    main()
