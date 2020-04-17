import logging
import os
import warnings
import pandas as pd
import pickle
from PIL import ImageFile
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from src.machine_learning.efficientnetb3 import EfficientNetConvInitializer, EfficientNetDenseInitializer, Swish
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
tf.compat.v1.disable_eager_execution()
ImageFile.LOAD_TRUNCATED_IMAGES = True

get_custom_objects()['EfficientNetConvInitializer'] = EfficientNetConvInitializer
get_custom_objects()['EfficientNetDenseInitializer'] = EfficientNetDenseInitializer
get_custom_objects()['Swish'] = Swish

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class MultitaskEncoder:
    """
    Uses the multitask model trained in the Categorizing the the three tasks
    """
    def __init__(self, algo, network, task, batch_size, min_el):
        self.algo = algo
        self.network = network
        self.task = task
        self.batch_size = batch_size
        self.min_el = min_el
        self.GENERATOR_DIMS = 112
        self.TARGET_DIMS = 112
        self.EPOCHS = 10

    def encode(self):
        """
        Reads the datasets, creates a generator for each one which feeds the trained models.
        The softmax layer of the models is removed

        :return: Embeddings for each of the datasets stored individually.
        """
        train = pd.read_csv(project_source + '/data/matching/train/train_' + str(self.min_el) + '.csv')
        valid = pd.read_csv(project_source + '/data/matching/valid/valid_' + str(self.min_el) + '.csv')
        test = pd.read_csv(project_source + '/data/matching/test/test_' + str(self.min_el) + '.csv')
        train = train.applymap(str)
        valid = valid.applymap(str)
        test = test.applymap(str)

        train['brand_id'] = train['brand_id'].apply(lambda el: 'b' + el)
        train['category_id'] = train['category_id'].apply(lambda el: 'c' + el)
        train['product_line_id'] = train['product_line_id'].apply(lambda el: 'p' + el)

        valid['brand_id'] = valid['brand_id'].apply(lambda el: 'b' + el)
        valid['category_id'] = valid['category_id'].apply(lambda el: 'c' + el)
        valid['product_line_id'] = valid['product_line_id'].apply(lambda el: 'p' + el)

        test['brand_id'] = test['brand_id'].apply(lambda el: 'b' + el)
        test['category_id'] = test['category_id'].apply(lambda el: 'c' + el)
        test['product_line_id'] = test['product_line_id'].apply(lambda el: 'p' + el)

        train['label'] = train.iloc[:, 1:-1].values.tolist()
        valid['label'] = valid.iloc[:, 1:-1].values.tolist()
        test['label'] = test.iloc[:, 1:-1].values.tolist()

        train_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
        valid_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')
        test_datagen = ImageDataGenerator(rescale=1. / 255, fill_mode='nearest')

        train_crops = train_datagen.flow_from_dataframe(train, x_col='id', y_col='label', interpolation='bicubic', class_mode='categorical',
                                                        classes=list(train['brand_id'].unique()) + list(train['category_id'].unique()) + list(train['product_line_id'].unique()),
                                                        target_size=(self.TARGET_DIMS, self.TARGET_DIMS), shuffle=True, batch_size=self.batch_size)
        valid_crops = valid_datagen.flow_from_dataframe(valid, x_col='id', y_col='label', interpolation='bicubic', class_mode='categorical',
                                                        classes=list(train['brand_id'].unique()) + list(train['category_id'].unique()) + list(train['product_line_id'].unique()),
                                                        target_size=(self.TARGET_DIMS, self.TARGET_DIMS), shuffle=False, batch_size=self.batch_size)
        test_crops = test_datagen.flow_from_dataframe(test, x_col='id', y_col='label', interpolation='bicubic', class_mode='categorical',
                                                      classes=list(train['brand_id'].unique()) + list(train['category_id'].unique()) + list(train['product_line_id'].unique()),
                                                      target_size=(self.TARGET_DIMS, self.TARGET_DIMS), shuffle=False, batch_size=self.batch_size)

        model = models.load_model(project_source + '/data/classification/models/' + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                                  str(self.min_el) + '.ckpt', compile=False)
        model = models.Model(inputs=model.input, outputs=model.get_layer("global_average_pooling2d").output)

        train_crops.reset()
        n_steps = (train_crops.n // train_crops.batch_size) + 1 if (train_crops.n % train_crops.batch_size) != 0 else (train_crops.n // train_crops.batch_size)
        embeddings = model.predict_generator(train_crops, steps=n_steps, verbose=1)
        res = pd.DataFrame({'embedding': list(embeddings), 'id': train['barcoded_product_id']})
        with open(project_source + '/data/matching/embeddings/train/' + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                  str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)

        valid_crops.reset()
        n_steps = (valid_crops.n // valid_crops.batch_size) + 1 if (valid_crops.n % valid_crops.batch_size) != 0 else (valid_crops.n // valid_crops.batch_size)
        embeddings = model.predict_generator(valid_crops, steps=n_steps, verbose=1)
        res = pd.DataFrame({'embedding': list(embeddings), 'id': valid['barcoded_product_id']})
        with open(project_source + '/data/matching/embeddings/valid/' + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                  str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)

        test_crops.reset()
        n_steps = (test_crops.n // test_crops.batch_size) + 1 if (test_crops.n % test_crops.batch_size) != 0 else (test_crops.n // test_crops.batch_size)
        embeddings = model.predict_generator(test_crops, steps=n_steps, verbose=1)
        res = pd.DataFrame({'embedding': list(embeddings), 'id': test['barcoded_product_id']})
        with open(project_source + '/data/matching/embeddings/test/' + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                  str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)
