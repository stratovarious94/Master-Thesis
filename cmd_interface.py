import cmd
import sys
import warnings
warnings.filterwarnings("ignore")

from src.feature_engineering.basic_pp import Preprocessor
from src.machine_learning.densenet import DenseNet
from src.machine_learning.inception_resnet import Inception_ResNet
from src.machine_learning.nasnet import NASNet
from src.machine_learning.efficientnetb3 import EfficientNetB3
from src.machine_learning.efficientnetb5 import EfficientNetB5
from src.evaluations.predict import Predictor
from src.statistics.dataset import show
from src.train.train_network import TrainNetwork
from src.train.train_multitask import TrainMultitask
from src.statistics.models import ModelStatistics
from src.statistics.matching import MatchingStatistics
from src.matching.encode import Encoder
from src.matching.encode_barcode import BarcodeEncoder
from src.matching.encode_multitask import MultitaskEncoder
from src.matching.match_simple import SimpleMatcher
from src.matching.match_complex import ComplexMatcher


class CommandLine(cmd.Cmd):
    """
    Interface controlling the functionality of the project. More information about this module in the README section
    """
    def __init__(self):
        cmd.Cmd.__init__(self)
        print("Input action of the form <action -0 -0 -0>. The arguments in order:\n"
              "Actions:: 0: Preprocess, 1: Train, 2: Evaluate, 3: Create Embeddings, 4: Perform Matching, 5: Dataset Statistics, 6: Model Statistics, 7: Matching Statistics\n"
              "Algorithms:: 0: Inception ResNet, 1: NasNet, 2: DenseNet, 3: Efficientnet B3, 4: Efficientnet B5\n"
              "Tasks:: 0: brand_id, 1: category_id, 2: product_line_id, 3: barcoded_product_id, 4: multitask\n"
              "Batch_size:: Integer \n"
              "Minimum number of elements per class:: Integer \n")

        self.action = ["preprocess", "train", "evaluate", "create_embeddings", "perform_matching", "dataset_statistics", "model_statistics", "matching_statistics"]
        self.algorithms = ["inception_", "nasnet_", "densenet", "efficientnetb3_", "efficientnetb5_"]
        self.networks = [Inception_ResNet, NASNet, DenseNet, EfficientNetB3, EfficientNetB5]
        self.tasks = ["brand_id", "category_id", "product_line_id", "barcoded_product_id", "multitask"]

    def preprocess(self, args):
        Preprocessor(int(args[1])).get_sets()

    def train(self, args):
        if int(args[2]) != 4:
            TrainNetwork(algo=self.algorithms[int(args[1])], network=self.networks[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).train()
        else:
            TrainMultitask(algo=self.algorithms[int(args[1])], network=self.networks[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).train()

    def evaluate(self, args):
        if int(args[2]) != 4:
            Predictor(algo=self.algorithms[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).predict()

    def encode(self, args):
        if int(args[2]) < 3:
            Encoder(algo=self.algorithms[int(args[1])], network=self.networks[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).encode()
        elif int(args[2]) == 3:
            BarcodeEncoder(algo=self.algorithms[int(args[1])], network=self.networks[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).encode()
        else:
            MultitaskEncoder(algo=self.algorithms[int(args[1])], network=self.networks[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).encode()

    def match(self, args):
        if int(args[2]) < 3:
            SimpleMatcher(algo=self.algorithms[int(args[1])], network=self.networks[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).match()
        else:
            ComplexMatcher(algo=self.algorithms[int(args[1])], network=self.networks[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).match()


    def dataset_statistics(self):
        show()

    def model_statistics(self, args):
        if int(args[2]) != 4:
            ModelStatistics(algo=self.algorithms[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).statistics()

    def matching_statistics(self, args):
        if int(args[2]) >= 3:
            MatchingStatistics(algo=self.algorithms[int(args[1])], task=self.tasks[int(args[2])], batch_size=int(args[3]), min_el=int(args[4])).statistics()

    def do_action(self, args=None):
        args = args.split("-")
        args = [arg.strip() for arg in args][1:]

        print(args)
        if self.action[int(args[0])] == "preprocess":
            self.preprocess(args)
        elif self.action[int(args[0])] == "train":
            self.train(args)
        elif self.action[int(args[0])] == "evaluate":
            self.evaluate(args)
        elif self.action[int(args[0])] == "create_embeddings":
            self.encode(args)
        elif self.action[int(args[0])] == "perform_matching":
            self.match(args)
        elif self.action[int(args[0])] == "dataset_statistics":
            self.dataset_statistics()
        elif self.action[int(args[0])] == "model_statistics":
            self.model_statistics(args)
        elif self.action[int(args[0])] == "matching_statistics":
            self.matching_statistics(args)
        else:
            print("Invalid action")

    def do_quit(self, args=None):
        sys.exit(int(args) if args else 0)


if __name__ == '__main__':
    CommandLine().cmdloop()
