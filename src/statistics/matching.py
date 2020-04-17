import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd

from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import classification_report

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 22})
tasks = ["brand_id", "category_id", "product_line_id"]

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class MatchingStatistics:
    """
    This class accepts the following arguments:
    algo: A string with the name of the algorithm being used
    task: A string with the name of the task the algorithm performs
    """
    def __init__(self, algo, task, batch_size, min_el):
        self.algo = algo
        self.task = task
        self.batch_size = batch_size
        self.min_el = min_el
        self.el = ['10', '50', '100']

    def load(self):
        """
        :returns: The history dump file
        """
        res_group = []
        datasets_group = []
        probabilities_group = []
        for el in self.el:
            res = []
            if Path(project_source + "/data/matching/predictions/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el).is_file():
                with (open(project_source + "/data/matching/predictions/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el, "rb")) as openfile:
                    while True:
                        try:
                            res.append(pickle.load(openfile))
                        except EOFError:
                            break
                res_group.append(res)

            prob = []
            if Path(project_source + "/data/classification/probabilities/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el).is_file():
                with (open(project_source + "/data/classification/probabilities/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el, "rb")) as openfile:
                    while True:
                        try:
                            prob.append(pickle.load(openfile))
                        except EOFError:
                            break
                probabilities_group.append(prob)

            datasets = []
            if Path(project_source + "/data/classification/train/train_" + el + '.csv').is_file():
                train = pd.read_csv(project_source + "/data/matching/train/train_" + el + '.csv')
                datasets.append(train)
            if Path(project_source + "/data/classification/valid/valid_" + el + '.csv').is_file():
                valid = pd.read_csv(project_source + "/data/matching/valid/valid_" + el + '.csv')
                datasets.append(valid)
            if Path(project_source + "/data/classification/test/test_" + el + '.csv').is_file():
                test = pd.read_csv(project_source + "/data/matching/test/test_" + el + '.csv')
                datasets.append(test)
                datasets_group.append(datasets)

        return [res_group, datasets_group, probabilities_group]

    def statistics(self):
        """
        Functionality:
        1) Plot the training and validation loss
        2) Plot the training and validation accuracy
        """
        res_group, datasets_group, probabilities_group = self.load()
        els = ['10', '50', '100']

        fig, ax = plt.subplots(figsize=(20, 10))
        ind = np.arange(3)
        for i, res in enumerate(res_group):
            ax.bar(ind[i], accuracy_score(res[0]['true'], res[0]['pred']), width=0.9, color='teal')
            ax.set(xlabel='Minimum Elements', ylabel='Accuracy', title='Accuracy Score for ' + self.task)
            ax.grid()
            ax.set_ylim(0, 1)
            ax.set_xticks(np.arange(3))
            ax.text(i - 0.118, accuracy_score(res[0]['true'], res[0]['pred']) + 0.005, str(accuracy_score(res[0]['true'], res[0]['pred']))[:5], color='black', fontweight='bold')
        ax.set_xticklabels([els[0], els[1], els[2]])
        plt.show()

        for i, res in enumerate(res_group):
            true = res[0]['true']
            pred = res[0]['pred']

            print('EfficientNet B5 - Minimum elements: ' + els[i] + '\n=============================================')

            # Accuracy Score
            print("Accuracy: ", accuracy_score(true, pred))

            # Precision Score
            print("Micro precision:", precision_score(true, pred, average="micro"))
            print("Macro precision: ", precision_score(true, pred, average="macro"))
            print("Weighted precision: ", precision_score(true, pred, average="weighted"))

            # Recall Score
            print("Micro recall: ", recall_score(true, pred, average="micro"))
            print("Macro recall: ", recall_score(true, pred, average="macro"))
            print("Weighted recall: ", recall_score(true, pred, average="weighted"))

            # F1 score
            print("Micro f1_score: ", f1_score(true, pred, average="micro"))
            print("Macro f1_score: ", f1_score(true, pred, average="macro"))
            print("Weighted f1_score: ", f1_score(true, pred, average="weighted"), '\n\n')

        for i, res in enumerate(res_group):
            fig, ax = plt.subplots(figsize=(20, 10))

            true = res[0]['true']
            pred = res[0]['pred']

            cr = classification_report(true, pred, target_names=list(true.unique()), output_dict=True)
            cr = pd.DataFrame(cr).transpose()[:-3]
            train = datasets_group[i][0]
            valid = datasets_group[i][1]
            test = datasets_group[i][2]
            incorrect = cr[cr['f1-score'] <= 0.8].index
            train_inc = train[train['barcoded_product_id'].isin(incorrect)]['barcoded_product_id'].value_counts()
            valid_inc = valid[valid['barcoded_product_id'].isin(incorrect)]['barcoded_product_id'].value_counts()
            test_inc = test[test['barcoded_product_id'].isin(incorrect)]['barcoded_product_id'].value_counts()

            g1 = cr['f1-score'].apply(lambda x: round(x, 1) * 100).value_counts().reset_index().sort_values('index')
            g1['index'] = g1['index'].apply(lambda el: str(int(el)) + '-' + str(int(el) + 9) if el != 100.0 else str(int(el)))
            ax.bar(g1['index'], g1['f1-score'])
            ax.set(xlabel='F1-score(%)', ylabel='Number of classes', title='F1-score in groups of 10. EfficientNet B5 - Minimum elements: ' + els[i])
            ax.grid()
            plt.show()

            bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
            groups = cr.groupby([pd.cut(cr['f1-score'], bins)])
            grouped_cr = groups.mean().dropna()

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.bar(np.arange(len(grouped_cr.index.values)), grouped_cr['support'].values, width=0.9, color='teal')
            ax.set(xlabel='F1-Score Group', ylabel='Mean number of products in the test set', title='Mean number of products per f1-score bucket in the test set.\n'
                                                                                                    'EfficientNet B5 - Minimum elements: ' + els[i])
            ax.grid()
            ax.set_xticks(np.arange(len(grouped_cr.index.to_list())))
            ax.set_xticklabels(grouped_cr.index.to_list())
            plt.show()
