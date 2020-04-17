import logging
import os
import pickle
import warnings
from PIL import ImageFile
warnings.filterwarnings("ignore")

import pandas as pd
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense

from src.optimizers.Adabound import AdaBoundOptimizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class TrainMultitask:
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
        self.EPOCHS = 15
        self.batch_size = batch_size
        self.algo = algo
        self.task = task
        self.network = network
        self.min_el = min_el

    def preprocess(self):
        """
        Read the training and validation sets, convert the labels in multitask format and count the total number of classes.

        :return: The datasets as dataframes
        """
        train = pd.read_csv(project_source + '/data/classification/train/train_' + str(self.min_el) + '.csv')[['id', 'brand_id', 'category_id', 'product_line_id']]
        valid = pd.read_csv(project_source + '/data/classification/valid/valid_' + str(self.min_el) + '.csv')[['id', 'brand_id', 'category_id', 'product_line_id']]

        train = train.applymap(str)
        valid = valid.applymap(str)

        train['brand_id'] = train['brand_id'].apply(lambda el: 'b' + el)
        train['category_id'] = train['category_id'].apply(lambda el: 'c' + el)
        train['product_line_id'] = train['product_line_id'].apply(lambda el: 'p' + el)

        valid['brand_id'] = valid['brand_id'].apply(lambda el: 'b' + el)
        valid['category_id'] = valid['category_id'].apply(lambda el: 'c' + el)
        valid['product_line_id'] = valid['product_line_id'].apply(lambda el: 'p' + el)

        train['label'] = train.iloc[:, 1:].values.tolist()
        valid['label'] = valid.iloc[:, 1:].values.tolist()

        print(train.shape)

        self.BRAND_ID_CLASSES = train['brand_id'].nunique()
        self.CATEGORY_ID_CLASSES = train['category_id'].nunique()
        self.PRODUCT_LINE_ID_CLASSES = train['product_line_id'].nunique()
        self.NUMBER_OF_CLASSES = train['brand_id'].nunique() + train['category_id'].nunique() + train['product_line_id'].nunique()

        self.STOP1 = self.BRAND_ID_CLASSES
        self.STOP2 = self.BRAND_ID_CLASSES + self.CATEGORY_ID_CLASSES

        return [train, valid]



    def generator_wrapper(self, generator):
        """
        Get the images yilded frok imagedatagenerator and convert them to multitask friendly format

        :param generator: The imagedatagenerator for training and validation set
        :return: The batches in multitask form
        """
        for batch_x, batch_y in generator:
            yield (batch_x, [batch_y[:, :self.STOP1], batch_y[:, self.STOP1:self.STOP2], batch_y[:, self.STOP2:]])

    def create_generators(self, train, valid):
        """
        :param train: The training set
        :param valid: The validation set
        :return: A generator for each set
        """
        train_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
        valid_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
        train_crops = train_datagen.flow_from_dataframe(train, x_col='id', y_col='label', interpolation='bicubic', class_mode='categorical',
                                                        classes=list(train['brand_id'].unique()) + list(train['category_id'].unique()) + list(train['product_line_id'].unique()),
                                                        target_size=(self.TARGET_DIMS, self.TARGET_DIMS), shuffle=True, batch_size=self.batch_size)
        valid_crops = valid_datagen.flow_from_dataframe(valid, x_col='id', y_col='label', interpolation='bicubic', class_mode='categorical',
                                                        classes=list(train['brand_id'].unique()) + list(train['category_id'].unique()) + list(train['product_line_id'].unique()),
                                                        target_size=(self.TARGET_DIMS, self.TARGET_DIMS), shuffle=False, batch_size=self.batch_size)

        return [train_crops, valid_crops]

    def create_model(self):
        """
        Reads the base model and changes the single softmax into three softmaxes for multitask purposes.

        :return: The multitask model
        """
        model = self.network(classes=self.NUMBER_OF_CLASSES, input=(self.TARGET_DIMS, self.TARGET_DIMS, self.NUMBER_OF_CHANNELS), weights='imagenet').create()
        x = model.get_layer("global_average_pooling2d").output
        output1 = Dense(self.BRAND_ID_CLASSES, activation='softmax')(x)
        output2 = Dense(self.CATEGORY_ID_CLASSES, activation='softmax')(x)
        output3 = Dense(self.PRODUCT_LINE_ID_CLASSES, activation='softmax')(x)
        model = models.Model(inputs=model.input, outputs=[output1, output2, output3])
        model.summary()
        model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], optimizer=AdaBoundOptimizer(), metrics=['accuracy'])
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
        cp_callback = ModelCheckpoint(filepath=project_source + '/data/classification/models/' + self.algo + self.task + '_' + str(self.batch_size) + '_' + str(self.min_el) + '.ckpt',
                                      monitor='val_dense_1_acc',
                                      verbose=0,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='auto',
                                      save_freq='epoch',
                                      load_weights_on_restart=False)

        history = model.fit_generator(self.generator_wrapper(train_crops),
                                      steps_per_epoch=(train.shape[0] // self.batch_size)+1,
                                      validation_data=self.generator_wrapper(valid_crops),
                                      validation_steps=(valid.shape[0] // self.batch_size)+1,
                                      epochs=self.EPOCHS,
                                      workers=1,
                                      max_queue_size=24,
                                      callbacks=[cp_callback])

        return history

    def save_history(self, history):
        """
        Saves the history for the training process

        :param history: Object containing the training history of the model
        """
        with open(project_source + '/data/classification/history/' + self.algo + self.task + '_' + str(self.batch_size) + '_' + str(self.min_el), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    def train(self):
        """
        Call the aforementioned functions in order
        """
        train, valid = self.preprocess()
        train_crops, valid_crops, = self.create_generators(train, valid)
        model = self.create_model()
        history = self.fit_model(model, train, valid, train_crops, valid_crops)
        self.save_history(history)
