from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from models.classifiers import ContentBasedClassifier


class RandomForest(ContentBasedClassifier):
    """
    Class to represent RandomForest model, extending the Classifier class
    """
    rf = None

    def train(self, properties, input_data, labels):
        """
        Train method for Random Forest classifier

        :param input_data: the training dataset
        :param labels: the training labels
        :param properties: properties from yaml
        """ 
        estimators = properties["rf"]["estimators"]
        max_depth = properties["rf"]["max_depth"]
        self.rf = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth, random_state=7)
        self.rf.fit(input_data, labels)

    def test(self, test_data, true_labels):
        """
        Method to test the Random Forest model

        :param test_data: testing dataset
        :param true_labels: testing labels
        :return: confusion matrix
        """
        predicted_labels = self.rf.predict(test_data)
        return confusion_matrix(true_labels, predicted_labels)
