from tensorflow.python.keras.applications.nasnet import NASNetLarge
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model


class NASNet:
    """
    This class accepts the following arguments:
    classes: Number of classes in which the softmax will apply probabilities
    input: The shape of the image which the neural network will recieve
    weights: Transfer Learning: Transfer the weights learned from an inception_resnet in imagenet
    """
    def __init__(self, classes=1000, input=(224, 224, 3), weights='imagenet'):
        self.classes = classes
        self.input = input
        self.weights = weights

    def create(self):
        """
        functionality:
        1) Create an inception_NASNet model
        2) Remove the top and add an AveragePooling and Softmax
        3) Unfreeze the top ~20 layers to allow specialization on the given dataset
        4) Print an overview of the model
        """
        model_base = NASNetLarge(include_top=False, input_shape=self.input, weights=self.weights)
        output = model_base.output
        output = GlobalAveragePooling2D(name='avg_pool')(output)
        output = Flatten()(output)
        output = Dense(self.classes, activation='softmax')(output)
        model = Model(model_base.input, output)
        for layer in model.layers[:-20]:
            layer.trainable = False
        print(len(model.layers))
        model.summary()
        return model
