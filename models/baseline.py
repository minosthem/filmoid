from models.classifiers import ContentBasedClassifier
import numpy as np


class Naive(ContentBasedClassifier):
    """
    Class representing a naive probabilistic model
    """
    models = []
    fold_metrics = []
    avg_metrics = {}
    test_metrics = {}
    best_model = None
    model_name = ""
    training_label_distribution = None

    def __init__(self):
        self.model_name = "naive"

    def train(self, properties, input_data, labels):
        """
        Train method for the majority classifier. Saves the training label distribution for classification.

        Args
            properties (dict): loaded from yaml file, uses the estimators and max_depth for RandomForest
            input_data (ndarray): the training set
            labels (ndarray): the labels of the training data
        """
        # store the training label distribution
        self.training_label_distribution = labels
        self.models.append(self.training_label_distribution)

    def test(self, test_data, true_labels, kind="validation"):
        """
        Method to test the Majority classifier

        Args
            test_data (ndarray): testing dataset
            true_labels (ndarray): testing labels
            kind (str): validation or test

        Returns
            confusion_matrix: the confusion matrix of the testing
        """
        # generate a sample of predictions, by sampling from the training label distribution
        tld = self.models[-1] if kind == "validation" else self.best_model
        predicted_labels = np.random.choice(tld, len(test_data))
        return true_labels, predicted_labels


class Random(ContentBasedClassifier):
    """
    Class that represents a random classifier.
    """
    models = []
    fold_metrics = []
    avg_metrics = {}
    test_metrics = {}
    best_model = None
    model_name = ""

    def __init__(self):
        self.model_name = "random"

    def train(self, properties, input_data, labels):
        """
        Train method for the random classifier.

        Args
            properties (dict): loaded from yaml file, uses the estimators and max_depth for RandomForest
            input_data (ndarray): the training set
            labels (ndarray): the labels of the training data
        """
        labelset = np.unique(labels)
        self.models.append(labelset)

    def test(self, test_data, true_labels, kind="validation"):
        """
        Method to test the random classifier

        Args
            test_data (ndarray): testing dataset
            true_labels (ndarray): testing labels
            kind (str): validation or test

        Returns
            confusion_matrix: the confusion matrix of the testing
        """
        tld = self.models[-1] if kind == "validation" else self.best_model
        predicted_labels = np.asarray([np.random.choice(tld) for _ in true_labels])
        return true_labels, predicted_labels
