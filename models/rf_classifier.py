from sklearn.ensemble import RandomForestClassifier

from models.classifiers import ContentBasedClassifier


class RandomForest(ContentBasedClassifier):
    """
    Class to represent RandomForest model, extending the Classifier class
    """

    def train(self, properties, input_data, labels):
        """
        Train method for Random Forest classifier

        :param input_data: the training dataset
        :param labels: the training labels
        :param properties: properties from yaml
        """
        estimators = properties["rf"]["estimators"]
        max_depth = properties["rf"]["max_depth"]
        rf = RandomForestClassifier(n_estimators=estimators, max_depth=max_depth, random_state=7)
        self.models.append(rf)
        rf.fit(input_data, labels)

    def test(self, test_data, true_labels, kind="validation"):
        """
        Method to test the Random Forest model

        Args
            test_data (ndarray): testing dataset
            true_labels (ndarray): testing labels
            kind (str): validation or test

        Returns
            confusion_matrix: the confusion matrix of the testing
        """
        predicted_labels = self.models[-1].predict(test_data) if kind == "validation" else self.best_model.predict(
            test_data)
        return true_labels, predicted_labels
