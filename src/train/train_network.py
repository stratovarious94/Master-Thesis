import logging
import os
import pickle
import warnings

import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint

from src.feature_extraction.cropgenerator import CropGenerator
from src.optimizers.Adabound import AdaBoundOptimizer

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class TrainNetwork:
    """
    for each task:
    1) Use preprocessing module to split the dataset
    2) Create a generator for each subset
    3) Create and compile the model
    4) Fit the model and save the model with the highest validation accuracy
    5) Save the history with information about the train process
    """
    def __init__(self, algo, network, task, batch_size, min_el):
        self.NUMBER_OF_CHANNELS = 3
        self.NUMBER_OF_CLASSES = None
        self.GENERATOR_DIMS = 112
        self.TARGET_DIMS = 112
        self.EPOCHS = 10
        self.batch_size = batch_size
        self.algo = algo
        self.network = network
        self.task = task
        self.min_el = min_el

    def preprocess(self):
        """
        Read the training and validation sets, convert the labels in multitask format and count the total number of classes.

        :return: The datasets as dataframes
        """
        train = pd.read_csv(project_source + '/data/classification/train/train_' + str(self.min_el) + '.csv')[['id', self.task]].rename(columns={self.task: 'label'})
        valid = pd.read_csv(project_source + '/data/classification/valid/valid_' + str(self.min_el) + '.csv')[['id', self.task]].rename(columns={self.task: 'label'})
        test = pd.read_csv(project_source + '/data/classification/test/test_' + str(self.min_el) + '.csv')[['id', self.task]].rename(columns={self.task: 'label'})
        train['label'] = train['label'].astype(str)
        valid['label'] = valid['label'].astype(str)
        test['label'] = test['label'].astype(str)

        self.NUMBER_OF_CLASSES = train['label'].value_counts().shape[0]

        return [train, valid, test]

    def create_generators(self, train, valid, test):
        """
        Get the images yilded frok imagedatagenerator and convert them to multitask friendly format

        :param generator: The imagedatagenerator for training and validation set
        :return: The batches in multitask form
        """
        crop_generator = CropGenerator(datatrain=train,
                                       datavalid=valid,
                                       datatest=test,
                                       width=self.GENERATOR_DIMS,
                                       height=self.GENERATOR_DIMS,
                                       cl=self.TARGET_DIMS,
                                       nclass=self.NUMBER_OF_CLASSES,
                                       epochs=self.EPOCHS,
                                       batch=self.batch_size)

        return crop_generator.generate_crops()

    def create_model(self):
        """
        Reads the base model and changes the single softmax into three softmaxes for multitask purposes.

        :return: The multitask model
        """
        model = self.network(classes=self.NUMBER_OF_CLASSES, input=(self.TARGET_DIMS, self.TARGET_DIMS,
                                                                    self.NUMBER_OF_CHANNELS), weights='imagenet')\
            .create()

        model.compile(loss="categorical_crossentropy", optimizer=AdaBoundOptimizer(), metrics=['acc'])
        return model

    def fit_model(self, model, train, valid, train_crops, valid_crops):
        """
        Commences the training process for the model and saves the best model in each iteration

        :param model: The multitask model
        :param train: The training set
        :param valid: The validation set
        :param train_crops: The imagedatagenerator for the training set
        :param valid_crops: The imagedatagenerator for the validation set
        :return: An object containing the training history for the model
        """
        cp_callback = ModelCheckpoint(filepath=project_source + '/data/classification/models/' + self.algo + self.task + '_' +
                                               str(self.batch_size) + '_' + str(self.min_el) + '.ckpt',
                                      monitor='val_acc',
                                      verbose=0,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='auto',
                                      save_freq='epoch',
                                      load_weights_on_restart=False)

        history = model.fit_generator(train_crops,
                                      steps_per_epoch=(train.shape[0] // self.batch_size)+1,
                                      validation_data=valid_crops,
                                      validation_steps=(valid.shape[0] // self.batch_size)+1,
                                      epochs=self.EPOCHS,
                                      workers=8,
                                      max_queue_size=24,
                                      callbacks=[cp_callback])

        return history

    def save_history(self, history):
        """
        Saves the history for the training process

        :param history: Object containing the training history of the model
        """
        with open(project_source + '/data/classification/history/' + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                  str(self.min_el), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def train(self):
        """
        Call the aforementioned functions in order
        """
        train, valid, test = self.preprocess()
        model = self.create_model()
        train_crops, valid_crops, test_crops = self.create_generators(train, valid, test)
        history = self.fit_model(model, train, valid, train_crops, valid_crops)
        self.save_history(history)
