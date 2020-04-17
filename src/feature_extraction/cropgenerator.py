import numpy as np
from PIL import ImageFile
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CropGenerator:
    """
    This class accepts the following arguments:
    dftrain: The path to the dataset with the URLs, their resolutions, and tasks
    dfvalid: The maximum resolution width allowed for images e.g. 3000 pixels
    dftest: The maximum resolution height allowed for images e.g. 3000 pixels
    IMAGE_SIZE: The resolution in which the images will be reshaped when entering the generator e.g. 353
    CROP_LENGTH: The resolution in which the images will be shortened after the cropping e.g. 331
    NUM_CLASSES: How many classes should the generator train the data for  e.g. 2000
    NUM_EPOCHS: In how many epochs should the generatr fit the data e.g. 50
    BATCH_SIZE: How many images will be fed to the generator per step in every epoch

    High level functionality:
    1) Retrieve dataset and remove images acording to specified characteristics
    2) Split the dataset in train, validation and test set
    3) Return the sunbsets and save the test set for future use
    """

    def __init__(self, datatrain, datavalid, datatest, width, height, cl, nclass, epochs, batch):
        """ Create a new point at the origin """
        self.dftrain = datatrain
        self.dfvalid = datavalid
        self.dftest = datatest
        self.IMAGE_SIZE = (width, height)
        self.CROP_LENGTH = cl
        self.NUM_CLASSES = nclass
        self.NUM_EPOCHS = epochs
        self.BATCH_SIZE = batch

    def random_crop(self, img, random_crop_size):
        """
        Random cropping algorithm
        """
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def crop_generator(self, batches, crop_length):
        """
        functionality:
        1) Get a batch yielded from the generator
        2) Transform it to np.array form
        3) Crop each image in the batch
        4) Yield it again in the format recieved
        """
        while True:
            batch_x, batch_y = next(batches)
            batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
            for i in range(batch_x.shape[0]):
                batch_crops[i] = self.random_crop(batch_x[i], (crop_length, crop_length))
            yield (batch_crops, batch_y)

    def train_generator(self):
        """
        Create an imagedatagenerator with:
        preprocess_input: The ability to convert Grayscale and RGBA into RGB, any input size into a target size
                          and transform RGB of scale 0-255 to 0-1 which performs better for ReLU type functions
        rotation_range: How much to rotate the image in pixels
        width_shift_range: Shift the image by at most 20% from the center horizontaly
        height_shift_range: Shift the image by at most 20% from the center vertically
        shear_range: Linear mapping that displaces the pixels in a fixed direction horizontaly
        zoom_range: Zoom in or out by 20%
        channel_shift_range: Change a bit the colors uniformly in the RGB spectrum
        horizontal_flip: Mirror image horizontaly witha chance

        flow_from_dataframe:
        Create a stream from the train set, shuffle it and train
        """
        train_datagen = ImageDataGenerator(rescale=1./255,
                                           # preprocessing_function=preprocess_input,
                                           # rotation_range=40,
                                           # width_shift_range=0.2,
                                           # height_shift_range=0.2,
                                           # shear_range=0.2,
                                           # zoom_range=0.2,
                                           # channel_shift_range=10,
                                           # horizontal_flip=True,
                                           fill_mode='nearest')

        return train_datagen.flow_from_dataframe(self.dftrain,
                                                 x_col='id',
                                                 y_col='label',
                                                 interpolation='bicubic',
                                                 class_mode='categorical',
                                                 classes=list(self.dftrain['label'].unique()),
                                                 target_size=self.IMAGE_SIZE,
                                                 shuffle=True,
                                                 batch_size=self.BATCH_SIZE)

    def validation_generator(self):
        """
        Create an imagedatagenerator with:
        Only basic preprocessing

        flow_from_dataframe:
        Create a stream from the validation set, don't shuffle it and train
        """
        valid_datagen = ImageDataGenerator(rescale=1./255, fill_mode='nearest')
        return valid_datagen.flow_from_dataframe(self.dfvalid,
                                                 x_col='id',
                                                 y_col='label',
                                                 interpolation='bicubic',
                                                 class_mode='categorical',
                                                 classes=list(self.dftrain['label'].unique()),
                                                 target_size=self.IMAGE_SIZE,
                                                 shuffle=False,
                                                 batch_size=self.BATCH_SIZE)

    def test_generator(self):
        """
        Create an imagedatagenerator with:
        Only basic preprocessing

        flow_from_dataframe:
        Create a stream from the test set, don't shuffle it and train
        """
        test_datagen = ImageDataGenerator(rescale=1./255, fill_mode='nearest')
        return test_datagen.flow_from_dataframe(self.dftest,
                                                x_col='id',
                                                y_col='label',
                                                interpolation='bicubic',
                                                class_mode=None,
                                                target_size=self.IMAGE_SIZE,
                                                shuffle=False,
                                                batch_size=self.BATCH_SIZE)

    def generate_crops(self):
        """
        Create the train, validation and test generators
        """
        # train_crops = self.crop_generator(self.train_generator(), self.CROP_LENGTH)
        # valid_crops = self.crop_generator(self.validation_generator(), self.CROP_LENGTH)
        # test_crops = self.crop_generator(self.test_generator(), self.CROP_LENGTH)
        train_crops = self.train_generator()
        valid_crops = self.validation_generator()
        test_crops = self.test_generator()
        return [train_crops, valid_crops, test_crops]

    def generate_test_crops(self):
        """
        This is a generator used only when evaluating the dataset
        """
        test_datagen = ImageDataGenerator(rescale=1./255, fill_mode='nearest') \
            .flow_from_dataframe(self.dftest,
                                 x_col='id',
                                 y_col='label',
                                 interpolation='bicubic',
                                 class_mode='categorical',
                                 target_size=self.IMAGE_SIZE,
                                 shuffle=False,
                                 batch_size=self.BATCH_SIZE)
        return test_datagen
