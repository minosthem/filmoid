from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from models.classifiers import ContentBasedClassifier


class KNN(ContentBasedClassifier):
    """
    KNN class models the K-Nearest-Neighbor classifier. Extends the Classifier class and implements the methods
    train and test using the KNN model.
    """

    def train(self, properties, input_data, labels):
        """
        Train method for KNN classifier
        :param input_data: the training dataset
        :param labels: the training labels
        :param properties: properties from yaml
        """
        knn = KNeighborsClassifier(n_neighbors=properties["knn"]["neighbors"])
        self.models.append(knn)
        knn.fit(input_data, labels)

    def test(self, test_data, true_labels, kind="validation"):
        """
        Method to test the KNN model

        Args
            test_data (ndarray): testing dataset
            true_labels (ndarray): testing labels
            kind (str): validation or test

        Returns
            confusion_matrix: the confusion matrix of the testing
        """
        predicted_labels = self.models[-1].predict(test_data) if kind == "validation" else self.best_model.predict(
            test_data)
        return confusion_matrix(true_labels, predicted_labels)
