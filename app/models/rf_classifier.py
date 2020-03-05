from sklearn.ensemble import RandomForestClassifier

from utils.enums import MetricKind, ContentBasedModels
from models.classifiers import ContentBasedClassifier


class RandomForest(ContentBasedClassifier):
    """
    Class to represent RandomForest model, extending the Classifier class
    """

    models = []
    fold_metrics = []
    avg_metrics = {}
    test_metrics = {}
    best_model = None
    model_name = ""

    def __init__(self):
        self.model_name = ContentBasedModels.rf.value

    def train(self, properties, input_data, labels):
        """
        Train method for Random Forest classifier. The model is created, trained and stored in a list in a class field.

        Args
            properties (dict): loaded from yaml file, uses the estimators and max_depth for RandomForest
            input_data (ndarray): the training set
            labels (ndarray): the labels of the training data
        """
        estimators = properties["rf"]["estimators"]
        max_depth = properties["rf"]["max_depth"]
        rf = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth, random_state=7)
        self.models.append(rf)
        rf.fit(input_data, labels)

    def test(self, test_data, true_labels, kind=MetricKind.validation.value):
        """
        Method to test the Random Forest model

        Args
            test_data (ndarray): testing dataset
            true_labels (ndarray): testing labels
            kind (str): validation or test

        Returns
            tuple: true and probabilities of predicted labels as nd arrays
        """
        predicted_labels = self.models[-1].predict_proba(test_data) if kind == MetricKind.validation.value else \
            self.best_model.predict_proba(test_data)
        return true_labels, predicted_labels
