from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def run_cross_validation(classifier_name, properties, input_data, labels, fold_idx):
    """
    Checks which classifier is selected and then takes the input data and labels which are divided into k folds.
    Each fold contains a tuple with the train and test indices. For every fold the model is trained and the confusion
    matrix is added to a list.

    :param classifier_name: the classifier which is selected
    :param properties: from properties yaml the value of the key neighbors
    :param input_data: the input data - movie vectors
    :param labels: true labels - ratings
    :param fold_idx: the number of folds
    :return: the classifier and the confusion matrices created for each fold
    """
    classifier = Classifier()
    if classifier_name == "knn":
        classifier = KNN()
    elif classifier_name == "rf":
        classifier = RandomForest()
    elif classifier_name == "dnn":
        classifier = DeepNN()
    matrices = []
    fold_idx = list(fold_idx)
    for idx, (train_index, test_index) in enumerate(fold_idx):
        print("Running fold #{}/{}".format(idx + 1, len(fold_idx)))
        input_train, input_test = input_data[train_index], input_data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        classifier.train(input_train, labels_train, properties)
        conf_matrix = classifier.test(input_test, labels_test)
        matrices.append(conf_matrix)
    return classifier, matrices


class Classifier:
    """
    Class to model a classifier. Each class extending the Classifier class should override the methods
    train and test.
    """

    def train(self, input_data, labels, properties):
        pass

    def test(self, test_data, true_labels):
        pass


class KNN(Classifier):
    """
    KNN class models the K-Nearest-Neighbor classifier. Extends the Classifier class and implements the methods
    train and test using the KNN model.
    """
    knn = None

    def train(self, input_data, labels, properties):
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


class RandomForest(Classifier):
    """
    Class to represent RandomForest model, extending the Classifier class
    """
    rf = None

    def train(self, input_data, labels, properties):
        """
        Train method for Random Forest classifier

        :param input_data: the training dataset
        :param labels: the training labels
        :param properties: properties from yaml
        """
        self.rf = RandomForestClassifier(random_state=7)
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


class DeepNN(Classifier):
    """
    Class representing a Deep Neural Network.
    """
    def train(self, input_data, labels, properties):
        pass

    def test(self, test_data, true_labels):
        pass
