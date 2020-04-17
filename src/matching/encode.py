import os
import warnings
import pandas as pd
import pickle
from tensorflow.python.keras import models
from src.feature_extraction.cropgenerator import CropGenerator

import tensorflow as tf
from src.machine_learning.efficientnetb3 import EfficientNetConvInitializer, EfficientNetDenseInitializer, Swish
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings("ignore")

get_custom_objects()['EfficientNetConvInitializer'] = EfficientNetConvInitializer
get_custom_objects()['EfficientNetDenseInitializer'] = EfficientNetDenseInitializer
get_custom_objects()['Swish'] = Swish

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class Encoder:
    """
    Uses the models trained in the Categorization tasks in order to produce image embeddings
    """
    def __init__(self, algo, network, task, batch_size, min_el):
        self.algo = algo
        self.network = network
        self.task = task
        self.batch_size = batch_size
        self.min_el = min_el
        self.GENERATOR_DIMS = 112
        self.TARGET_DIMS = 112
        self.EPOCHS = 30

    def encode(self):
        """
        Reads the datasets, creates a generator for each one which feeds the trained models.
        The softmax layer of the models is removed

        :return: Embeddings for each of the datasets stored individually.
        """
        train = pd.read_csv(project_source + '/data/matching/train/train_' + str(self.min_el) +
                            '.csv')[['id', self.task, 'barcoded_product_id']].rename(columns={self.task: 'label'})
        valid = pd.read_csv(project_source + '/data/matching/valid/valid_' + str(self.min_el) +
                            '.csv')[['id', self.task, 'barcoded_product_id']].rename(columns={self.task: 'label'})
        test = pd.read_csv(project_source + '/data/matching/test/test_' + str(self.min_el) +
                           '.csv')[['id', self.task, 'barcoded_product_id']].rename(columns={self.task: 'label'})

        train['label'] = train['label'].astype(str)
        valid['label'] = valid['label'].astype(str)
        test['label'] = test['label'].astype(str)

        cropgenerator= CropGenerator(datatrain=train, datavalid=valid, datatest=test, width=self.GENERATOR_DIMS,
                                     height=self.GENERATOR_DIMS, cl=self.TARGET_DIMS,
                                     nclass=test['label'].value_counts().shape[0], epochs=self.EPOCHS,
                                     batch=self.batch_size)

        train_crops, valid_crops, test_crops = cropgenerator.generate_crops()

        model = models.load_model(project_source + '/data/classification/models/' + self.algo + self.task + '_' + str(self.batch_size) + '_' +
                                  str(self.min_el) + '.ckpt', compile=False)
        model.summary()
        model = models.Model(inputs=model.input, outputs=model.get_layer("global_average_pooling2d").output)
        model.summary()

        train_crops.reset()
        n_steps = (train_crops.n // train_crops.batch_size) + 1 if (train_crops.n % train_crops.batch_size) != 0 else (train_crops.n // train_crops.batch_size)
        embeddings = model.predict_generator(train_crops, steps=n_steps, verbose=1)
        res = pd.DataFrame({'embedding': list(embeddings), 'id': train['barcoded_product_id']})
        with open(project_source + '/data/matching/embeddings/train/' + self.algo + self.task + '_' + str(self.batch_size) + '_' + str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)

        valid_crops.reset()
        n_steps = (valid_crops.n // valid_crops.batch_size) + 1 if (valid_crops.n % valid_crops.batch_size) != 0 else (valid_crops.n // valid_crops.batch_size)
        embeddings = model.predict_generator(valid_crops, steps=n_steps, verbose=1)
        res = pd.DataFrame({'embedding': list(embeddings), 'id': valid['barcoded_product_id']})
        with open(project_source + '/data/matching/embeddings/valid/' + self.algo + self.task + '_' + str(self.batch_size) + '_' + str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)

        test_crops.reset()
        n_steps = (test_crops.n // test_crops.batch_size) + 1 if (test_crops.n % test_crops.batch_size) != 0 else (test_crops.n // test_crops.batch_size)
        embeddings = model.predict_generator(test_crops, steps=n_steps, verbose=1)
        res = pd.DataFrame({'embedding': list(embeddings), 'id': test['barcoded_product_id']})
        with open(project_source + '/data/matching/embeddings/test/' + self.algo + self.task + '_' + str(self.batch_size) + '_' + str(self.min_el), 'wb') as file_pi:
            pickle.dump(res, file_pi)



