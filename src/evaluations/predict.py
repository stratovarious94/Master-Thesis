import warnings
import numpy as np
import os
import pandas as pd
import pickle
from src.evaluations import helpers
from src.machine_learning.efficientnetb3 import EfficientNetConvInitializer, EfficientNetDenseInitializer, Swish

import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")

get_custom_objects()['EfficientNetConvInitializer'] = EfficientNetConvInitializer
get_custom_objects()['EfficientNetDenseInitializer'] = EfficientNetDenseInitializer
get_custom_objects()['Swish'] = Swish

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class Predictor:
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

    def load(self):
        """
        Functionality:
        1) Read the test set and the model
        2) Use the test set to create the generator
        :return: All the above
        """
        test_set = helpers.read_test_dataframes(self.algo, self.task, self.batch_size, self.min_el)
        model = helpers.read_model(self.algo, self.task, self.batch_size, self.min_el)
        test_generator = helpers.create_test_crops(test_set, self.batch_size)
        return [test_set, model, test_generator]

    def predict(self):
        """
        Functionality:
        1) Use the generator's evaluate_generator method to get the accuracy in the test set
        2) Store the results as a dump
        """
        test_set, model, test_generator = self.load()

        print(test_set.shape)

        test_generator.reset()
        n_steps = (test_generator.n // test_generator.batch_size) + 1 if (test_generator.n % test_generator.batch_size) != 0 else (test_generator.n // test_generator.batch_size)
        pred = model.predict_generator(test_generator, steps=n_steps, verbose=1)

        res = pd.DataFrame({'true': test_set['label'].astype(int), 'pred': list(pred)})

        with open(project_source + '/data/classification/probabilities/' + self.algo + self.task + '_' + str(self.batch_size) + '_' + str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)

        predicted_class_indices = np.argmax(pred, axis=1)
        labels = test_generator.class_indices
        labels = dict((v, k) for k, v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]

        res = pd.DataFrame({'true': test_set['label'].astype(int), 'pred': np.array(predictions).astype(int)})

        with open(project_source + '/data/classification/predictions/' + self.algo + self.task + '_' + str(self.batch_size) + '_' + str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)

#9NyPQu3C4ReNDBumdA2D