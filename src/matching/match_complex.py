import os
import numpy as np
import pandas as pd
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class ComplexMatcher:
    """
    This class is performing product matching for the multitask model
    """
    def __init__(self, algo, network, task, batch_size, min_el):
        self.algo = algo
        self.network = network
        self.task = task
        self.batch_size = batch_size
        self.min_el = min_el
        self.tasks = ['brand_id', 'category_id', 'product_line_id']

    def match(self):
        """
        This function reads the embeddings produced from the multitask model encoding process
        and performs KNN to find the most similar products

        :return: A pickle file containing the results of the matching process
        """
        train = []
        with (open(project_source + "/data/matching/embeddings/train/" + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                   str(self.min_el), "rb")) as openfile:
            while True:
                try:
                    train.append(pickle.load(openfile))
                except EOFError:
                    break
        train_x = np.stack(train[0]['embedding'].values)
        train_y = train[0]['id'].values

        test = []
        with (open(project_source + "/data/matching/embeddings/test/" + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                   str(self.min_el), "rb")) as openfile:
            while True:
                try:
                    test.append(pickle.load(openfile))
                except EOFError:
                    break
        test_x = np.stack(train[0]['embedding'].values)
        test_y = train[0]['id'].values

        print('Matching...')
        clf = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=None)

        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)

        print('accuracy: ', accuracy_score(test_y, pred))
        res = pd.DataFrame({'true': test_y, 'pred': np.array(pred)})
        with open(project_source + '/data/matching/predictions/' + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                  str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)
