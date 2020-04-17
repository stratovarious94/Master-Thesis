import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd

from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc, roc_curve, classification_report
from sklearn.preprocessing import label_binarize
from scipy import interp

tasks = ["brand_id", "category_id", "product_line_id", 'multitask']

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class ModelStatistics:
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
        objects_group = []
        res_group = []
        datasets_group = []
        probabilities_group = []
        for el in self.el:
            objects = []
            if Path(project_source + "/data/classification/history/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el).is_file():
                with (open(project_source + "/data/classification/history/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el, "rb")) as openfile:
                    while True:
                        try:
                            objects.append(pickle.load(openfile))
                        except EOFError:
                            break
                objects_group.append(objects)

            res = []
            if Path(project_source + "/data/classification/predictions/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el).is_file():
                with (open(project_source + "/data/classification/predictions/" + self.algo + self.task + '_' + str(self.batch_size) + '_' + el, "rb")) as openfile:
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
                train = pd.read_csv(project_source + "/data/classification/train/train_" + el + '.csv')
                datasets.append(train)
            if Path(project_source + "/data/classification/valid/valid_" + el + '.csv').is_file():
                valid = pd.read_csv(project_source + "/data/classification/valid/valid_" + el + '.csv')
                datasets.append(valid)
            if Path(project_source + "/data/classification/test/test_" + el + '.csv').is_file():
                test = pd.read_csv(project_source + "/data/classification/test/test_" + el + '.csv')
                datasets.append(test)
                datasets_group.append(datasets)

        return [objects_group, res_group, datasets_group, probabilities_group]

    def statistics(self):
        """
        Functionality:
        1) Plot the training and validation loss
        2) Plot the training and validation accuracy
        3) Plot the barplot for the three setting (10, 50, 100)
        4) Print micro, macro, weighted accuracy and f1 scores for each setting
        5) Plot F1-score in groups of 10
        6) Plot all classes with f1-score<=0.7
        7) Plot mean number of products per f1-score bucket in the test set
        8) Plot probability distribution (Confidence) in groups of 10
        9) Plot micro-average ROC curve
        10) Plot PR curve
        """
        objects_group, res_group, datasets_group, probabilities_group = self.load()
        els = ['10', '50', '100']

        # # Training loss plot
        for i, objects in enumerate(objects_group):
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(np.arange(1, len(objects[0]["acc"])+1, 1), objects[0]["loss"], label='Training set', linewidth=4)
            ax.plot(np.arange(1, len(objects[0]["acc"])+1, 1), objects[0]["val_loss"], label='Test set', linewidth=4)
            ax.set(xlabel='Epochs', ylabel='Cross entropy loss', title='Loss for the training process for ' + self.task + '. At least: ' + els[i] + ' images per class')
            ax.grid()
            ax.legend()
            ax.set_ylim(0, 1)
            ax.set_xticks(np.arange(1, len(objects[0]["acc"])+1, 1))
            plt.show()

        # Training accuracy plot
        for i, objects in enumerate(objects_group):
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(np.arange(1, len(objects[0]["acc"])+1, 1), objects[0]["acc"], label='Training set', linewidth=4)
            ax.plot(np.arange(1, len(objects[0]["acc"])+1, 1), objects[0]["val_acc"], label='Test set', linewidth=4)
            ax.set_title('Task: ' + self.task[:-3] + '. At least ' + els[i] + ' images per class\n', fontdict={'fontsize': 36})
            ax.set_xlabel('Epochs', fontdict={'fontsize': 36})
            ax.set_ylabel('Accuracy', fontdict={'fontsize': 36})
            ax.set_ylim(0, 1)
            ax.legend(prop={'size': 30})
            ax.set_ylim(0, 1)
            ax.grid()
            ax.set_xticks(np.arange(start=1, stop=len(objects[0]["acc"])+1))
            ax.tick_params(axis='both', which='major', labelsize=24)
            plt.show()

        fig, ax = plt.subplots(figsize=(20, 10))
        ind = np.arange(3)
        for i, res in enumerate(res_group):
            print(res[0]['true'])
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
            train_inc = train[train[self.task].isin(incorrect)][self.task].value_counts()
            valid_inc = valid[valid[self.task].isin(incorrect)][self.task].value_counts()
            test_inc = test[test[self.task].isin(incorrect)][self.task].value_counts()

            # Print F1 score in buckets of 10%
            g1 = cr['f1-score'].apply(lambda x: round(x, 1) * 100).value_counts().reset_index().sort_values('index')
            g1['index'] = g1['index'].apply(lambda el: str(int(el)) + '-' + str(int(el) + 9) if el != 100.0 else str(int(el)))
            ax.bar(g1['index'], g1['f1-score'])
            ax.set(xlabel='F1-score(%)', ylabel='Number of classes', title='F1-score in groups of 10. EfficientNet B5 - Minimum elements: ' + els[i])
            ax.grid()
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            axtest = ax.bar(np.arange(len(test_inc.index.to_list())), train_inc.values + valid_inc.values + test_inc.values, width=0.9, color='green')
            axvalid = ax.bar(np.arange(len(valid_inc.index.to_list())), train_inc.values + valid_inc.values, width=0.9, color='orange')
            axtrain = ax.bar(np.arange(len(train_inc.index.to_list())), train_inc.values, width=0.9, color='blue')

            ax.set(xlabel='Number of items in the 3 sets', ylabel='Number of entries', title='Classes with f1-score<=0.7. EfficientNet B5 - Minimum elements: ' + els[i])
            ax.grid()
            ax.set_xticks(np.arange(len(train_inc.index.to_list())))
            ax.set_xticklabels(train_inc.index.to_list())
            ax.legend((axtrain, axvalid, axtest), ('training set', 'validation set', 'test set'))
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

        for i, prob in enumerate(probabilities_group):
            fig, ax = plt.subplots(figsize=(20, 10))
            true = res_group[i][0]['true']
            pred = res_group[i][0]['pred']
            prob[0]['dec'] = np.equal(true, pred)
            max_prob = prob[0]['pred'].apply(lambda x: round(np.max(x), 1)).value_counts().reset_index().sort_values('index')
            max_prob['index'] = max_prob['index'].apply(lambda el: str(round(el - 0.05, 2)) + '-' + str(round(el + 0.05, 2)) if el != 1 else str(round(el - 0.05, 2)) + '-' + str(el))
            if len(max_prob) == 11:
                max_prob['index'].iloc[0] = '0.00' + max_prob['index'].iloc[0][3:]
            ax.bar(max_prob['index'], max_prob['pred'])
            ax.set(xlabel='Number of products', ylabel='Probability', title='Probability Distribution in groups of 10. EfficientNet B5 - Minimum elements: ' + els[i])
            ax.grid()
            ax.set_xticklabels(max_prob['index'], rotation=30)
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            max_prob = prob[0][prob[0]['dec'] == True]['pred'].apply(lambda x: round(np.max(x), 1)).value_counts().reset_index().sort_values('index')
            max_prob['index'] = max_prob['index'].apply(lambda el: str(round(el - 0.05, 2)) + '-' + str(round(el + 0.05, 2)) if el != 1 else str(round(el - 0.05, 2)) + '-' + str(el))
            if len(max_prob) == 11:
                max_prob['index'].iloc[0] = '0.00' + max_prob['index'].iloc[0][3:]
            ax.bar(max_prob['index'], max_prob['pred'])
            ax.set(xlabel='Number of products', ylabel='Probability', title='Probability Distribution in groups of 10. EfficientNet B5 - Minimum elements: ' + els[i])
            ax.grid()
            ax.set_xticklabels(max_prob['index'], rotation=30)
            plt.show()

            fig, ax = plt.subplots(figsize=(20, 10))
            max_prob = prob[0][prob[0]['dec'] == False]['pred'].apply(lambda x: round(np.max(x), 1)).value_counts().reset_index().sort_values('index')
            max_prob['index'] = max_prob['index'].apply(lambda el: str(round(el - 0.05, 2)) + '-' + str(round(el + 0.05, 2)) if el != 1 else str(round(el - 0.05, 2)) + '-' + str(el))
            if len(max_prob) == 11:
                max_prob['index'].iloc[0] = '0.00' + max_prob['index'].iloc[0][3:]
            ax.bar(max_prob['index'], max_prob['pred'])
            ax.set(xlabel='Number of products', ylabel='Probability', title='Probability Distribution in groups of 10. EfficientNet B5 - Minimum elements: ' + els[i])
            ax.grid()
            ax.set_xticklabels(max_prob['index'], rotation=30)
            plt.show()

        for k, res in enumerate(res_group):
            true = res[0]['true']
            pred = res[0]['pred']

            true_bin = label_binarize(true, classes=true.unique())
            pred_bin = label_binarize(pred, classes=true.unique())

            # Macro and micro ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(true_bin.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(true_bin[:, i], pred_bin[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            fpr["micro"], tpr["micro"], _ = roc_curve(true_bin.ravel(), pred_bin.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(true_bin.shape[1])]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(true_bin.shape[1]):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= true_bin.shape[1]
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            fig, ax = plt.subplots(figsize=(20, 10))
            plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='deeppink', linestyle='-', linewidth=4)
            plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]), color='navy', linestyle='-', linewidth=4)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Average ROC for all dataset. EfficientNet B5 - Minimum elements: ' + els[k])
            plt.legend(loc="best", fontsize=14)
            plt.show()

            # Precision Recall Curve
            precision = dict()
            recall = dict()
            average_precision = dict()

            for i in range(true_bin.shape[1]):
                precision[i], recall[i], _ = precision_recall_curve(true_bin[:, i], pred_bin[:, i])
                average_precision[i] = average_precision_score(true_bin[:, i], pred_bin[:, i])

            precision["micro"], recall["micro"], _ = precision_recall_curve(true_bin.ravel(), pred_bin.ravel())
            average_precision["micro"] = average_precision_score(true_bin, pred_bin, average="micro")

            fig, ax = plt.subplots(figsize=(20, 10))
            ax.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
            ax.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')
            ax.set(xlabel='Recall', ylabel='Precision', title=('Average precision score, micro-averaged over all '
                                                               'classes: AP={0:0.2f}. EfficientNet B5 - Minimum '
                                                               'elements: ' + els[k])
                   .format(average_precision["micro"]))
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            plt.show()
