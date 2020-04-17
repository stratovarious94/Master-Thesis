import os

from src.optimizers.Adabound import AdaBoundOptimizer
from src.feature_extraction.cropgenerator import CropGenerator

import pandas as pd
from tensorflow.python.keras import models

MAX_DIMS = 3000
NUMBER_OF_CHANNELS = 3
NOT_USED_NUMBER_OF_CHANNELS = 2
NUMBER_OF_CLASSES = 2526
GENERATOR_DIMS = 112
TARGET_DIMS = 112
EPOCHS = 30

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


def read_test_dataframes(algo, task, batch_size, min_el):
    """
    Utility function that reads the a specific dataset given:

    :param algo: Name of the model that has been used in the dataset
    :param task: Name of the task which the model has performed
    :param batch_size: The batch size with which the model has operated
    :param min_el: The minimum number of images per class allowed for the dataset
    :return: The dataframe created from reading specified file
    """
    test_set = pd.read_csv(project_source + '/data/classification/test/test_' + str(min_el) + '.csv')[['id', task]].rename(columns={task: 'label'})
    test_set['label'] = test_set['label'].astype(str)
    return test_set


def read_model(algo, task, batch_size, min_el):
    """
    Utility function that reads and returns the weights of a specific model given:

    :param algo: Name of the trained model
    :param task: Name of the task in which the model has been trained
    :param batch_size: The batch size with which the model has operated
    :param min_el: The minimum number of images per class allowed for the dataset in which the model was trained
    :return: The weights of the model specified
    """
    model = models.load_model(project_source + '/data/classification/models/' + algo + task + '_' + str(batch_size) + '_' + str(min_el) + '.ckpt', compile=False)
    model.compile(loss="categorical_crossentropy", optimizer=AdaBoundOptimizer(), metrics=['acc'])
    return model


def create_test_crops(test_sets, batch_size):
    """
    Utility function that accepts a dataframe and creates an imagedatagenerator using it

    :param test_sets: The dataframe which will be fed to the imagedatagenerator
    :param batch_size: How many images per batch will the imagedatagenerator utilize
    :return: The imagedatagenerator object
    """
    cropgenerator = CropGenerator(datatrain=None,
                                  datavalid=None,
                                  datatest=test_sets,
                                  width=GENERATOR_DIMS,
                                  height=GENERATOR_DIMS,
                                  cl=TARGET_DIMS,
                                  nclass=test_sets['label'].value_counts().shape[0],
                                  epochs=EPOCHS,
                                  batch=batch_size)

    test_crops = cropgenerator.generate_test_crops()
    return test_crops
