import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic
import warnings

from featurizer import Featurizer
from model_selection import ModelSelection

warnings.filterwarnings("ignore")


class Classifier:
    """The Classifier"""

    def __init__(self):
        self.dataset_columns = [
            "polarity",
            "aspect_category",
            "target_term",
            "character_offsets",
            "sentence",
        ]
        self.featurizer = Featurizer()
        kernel = ConstantKernel() * RationalQuadratic()
        self.model = GaussianProcessClassifier(kernel)

    def train(self, trainfile, devfile=None, select_model: bool = False):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_set = pd.read_csv(trainfile, sep="\t", names=self.dataset_columns)
        features, labels = self.featurizer.get_features_and_labels(
            train_set, train=True
        )
        if select_model:
            dev_set = pd.read_csv(devfile, sep="\t", names=self.dataset_columns)
            dev_features, dev_labels = self.featurizer.get_features_and_labels(
                dev_set, train=False
            )
            model_selection = ModelSelection(features, labels, dev_features, dev_labels)
            self.model = model_selection.model_tuning(self.model)
        self.model.fit(features, labels)

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        test_set = pd.read_csv(datafile, sep="\t", names=self.dataset_columns)
        features, labels = self.featurizer.get_features_and_labels(
            test_set, train=False
        )
        predictions = self.model.predict(features)
        polarities = self.featurizer.polarity_label_encoder.inverse_transform(
            predictions
        )
        return polarities
