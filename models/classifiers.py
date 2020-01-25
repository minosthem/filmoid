from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def run_cross_validation(classifier_name, properties, input, labels, fold_idx):
    """
    Checks which classifier is selected and then takes the input data and labels which are divided into k folds.
    Each fold contains a tuple with the train and test indices. For every fold the model is trained and the confusion
    matrix is added to a list.

    :param classifier_name: the classifier which is selected
    :param properties: from properties yaml the value of the key neighbors
    :param input: the input data - movie vectors
    :param labels: true labels - ratings
    :param fold_idx: the number of folds
    :return: the classifier and the confusion matrices created for each fold
    """
    classifier = Classifier()
    if classifier_name == "knn":
        classifier = KNN()
        classifier.neighbors = properties["neighbors"]
    elif classifier_name == "rf":
        classifier = RandomForest()
    elif classifier_name == "dnn":
        classifier = DeepNN()
    matrices = []
    fold_idx = list(fold_idx)
    for idx, (train_index, test_index) in enumerate(fold_idx):
        print("Running fold #{}/{}".format(idx+1, len(fold_idx)))
        input_train, input_test = input[train_index], input[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        classifier.train(input_train, labels_train)
        conf_matrix = classifier.test(input_test, labels_test)
        matrices.append(conf_matrix)
    return classifier, matrices


def run_test(classifier, test_data, test_labels):
    return classifier.test(test_data, test_labels)


class Classifier:

    def train(self, input, labels):
        pass

    def test(self, test, true_labels):
        pass


class KNN(Classifier):
    neighbors = 0
    knn = None

    def train(self, input, labels):
        self.knn = KNeighborsClassifier(n_neighbors=self.neighbors)
        self.knn.fit(input, labels)

    def test(self, test, true_labels):
        predicted_labels = self.knn.predict(test)
        return confusion_matrix(true_labels, predicted_labels)


class RandomForest(Classifier):

    def train(self, input, labels):
        pass

    def test(self, test, true_labels):
        pass


class DeepNN(Classifier):

    def train(self, input, labels):
        pass

    def test(self, test, true_labels):
        pass
