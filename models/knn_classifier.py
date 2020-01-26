from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from models.classifiers import ContentBasedClassifier


class KNN(ContentBasedClassifier):
    """
    KNN class models the K-Nearest-Neighbor classifier. Extends the Classifier class and implements the methods
    train and test using the KNN model.
    """
    knn = None

    def train(self, properties, input_data, labels):
        """
        Train method for KNN classifier
        :param input_data: the training dataset
        :param labels: the training labels
        :param properties: properties from yaml
        """
        self.knn = KNeighborsClassifier(n_neighbors=properties["knn"]["neighbors"])
        self.knn.fit(input_data, labels)

    def test(self, test_data, true_labels):
        """
        Method to test the KNN model
        :param test_data: testing dataset
        :param true_labels: testing labels
        :return: confusion matrix
        """
        predicted_labels = self.knn.predict(test_data)
        return confusion_matrix(true_labels, predicted_labels)
