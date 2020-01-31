import numpy as np


class Classifier:
    """
    Class to model a classifier. Each class extending the Classifier class should override the methods
    train and test.
    """

    def train(self, properties, input_data, labels):
        pass

    def test(self, test_data, true_labels, kind="validation"):
        pass


class ContentBasedClassifier(Classifier):
    """
    Class to be extended by content-based models. Methods train and test should be implemented by the
    sub-classes.
    """
    models = []
    best_model = None

    def train(self, properties, input_data, labels):
        pass

    def test(self, test_data, true_labels, kind="validation"):
        pass

    @staticmethod
    def run_cross_validation(classifier, properties, input_data, labels, fold_idx):
        """
        Checks which classifier is selected and then takes the input data and labels which are divided into k folds.
        Each fold contains a tuple with the train and test indices. For every fold the model is trained and the
        confusion matrix is added to a list.

        :param classifier: the classifier which is selected
        :param properties: from properties yaml the value of the key neighbors
        :param input_data: the input data - movie vectors
        :param labels: true labels - ratings
        :param fold_idx: the number of folds
        :return: the classifier and the confusion matrices created for each fold
        """

        matrices = []
        labels = np.asarray(labels)
        for idx, (train_index, test_index) in enumerate(fold_idx):
            print("Running fold #{}/{}".format(idx + 1, len(fold_idx)))
            input_train, input_test = input_data[train_index], input_data[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            classifier.train(properties=properties, input_data=input_train, labels=labels_train)
            conf_matrix = classifier.test(test_data=input_test, true_labels=labels_test)
            matrices.append(conf_matrix)
        return classifier, matrices



