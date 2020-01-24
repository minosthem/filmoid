from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


def run_cross_validation(classifier_name, properties, input, labels, fold_idx):
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
        conf_matrix = classifier.evaluate(input_test, labels_test)
        matrices.append(conf_matrix)
    return classifier, matrices


def run_test(classifier, test_data, test_labels):
    pass


class Classifier:

    def train(self, input, labels):
        pass

    def evaluate(self, test, true_labels):
        pass

    def test(self, test_data, test_labels):
        pass


class KNN(Classifier):
    neighbors = 0
    knn = None

    def train(self, input, labels):
        self.knn = KNeighborsClassifier(n_neighbors=self.neighbors)
        self.knn.fit(input, labels)

    def evaluate(self, test, true_labels):
        predicted_labels = self.knn.predict(test)
        return confusion_matrix(true_labels, predicted_labels)

    def test(self, test_data, test_labels):
        pass


class RandomForest(Classifier):

    def train(self, input, labels):
        pass

    def evaluate(self, test, true_labels):
        pass

    def test(self, test_data, test_labels):
        pass


class DeepNN(Classifier):

    def train(self, input, labels):
        pass

    def evaluate(self, test, true_labels):
        pass

    def test(self, test_data, test_labels):
        pass
