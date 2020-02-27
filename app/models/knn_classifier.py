from sklearn.neighbors import KNeighborsClassifier

from utils.enums import MetricKind, ContentBasedModels
from models.classifiers import ContentBasedClassifier


class KNN(ContentBasedClassifier):
    """
    KNN class models the K-Nearest-Neighbor classifier. Extends the Classifier class and implements the methods
    train and test using the KNN model.
    """

    models = []
    fold_metrics = []
    avg_metrics = {}
    test_metrics = {}
    best_model = None
    model_name = ""

    def __init__(self):
        self.model_name = ContentBasedModels.knn.value

    def train(self, properties, input_data, labels):
        """
        Train method for KNN classifier. Creates a new KNN model and stores it in a list. The model is trained
        using the input_data and labels.

        Args
            properties (dict): loaded from yaml file. Uses the property knn-neighbors to define the number of neighbors
            for the model.
            input_data (ndarray): the training set
            labels (ndarray): the labels of the training data
        """
        knn = KNeighborsClassifier(n_neighbors=properties["knn"]["neighbors"])
        self.models.append(knn)
        knn.fit(input_data, labels)

    def test(self, test_data, true_labels, kind=MetricKind.validation.value):
        """
        Method to test the KNN model

        Args
            test_data (ndarray): testing dataset
            true_labels (ndarray): testing labels
            kind (str): validation or test

        Returns
            tuple: true and predicted labels as nd arrays
        """
        predicted_labels = self.models[-1].predict(test_data) if kind == MetricKind.validation.value else \
            self.best_model.predict(test_data)
        return true_labels, predicted_labels
