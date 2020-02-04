from os import mkdir
from os.path import join, exists
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

import utils
from enums import MetricNames, MetricKind
from utils import logger


class Classifier:
    """
    Class to model a classifier. Each class extending the Classifier class should override the methods
    train and test.
    """

    def train(self, properties, input_data, labels):
        pass

    def test(self, test_data, true_labels, kind=MetricKind.validation.value):
        pass


class ContentBasedClassifier(Classifier):
    """
    Class to be extended by content-based models. Methods train and test should be implemented by the
    sub-classes.
    """
    models = []
    fold_metrics = []
    avg_metrics = {}
    test_metrics = {}
    best_model = None
    model_name = ""

    def train(self, properties, input_data, labels):
        pass

    def test(self, test_data, true_labels, kind=MetricKind.validation.value):
        pass

    def get_results(self, true_labels, predicted_labels, kind=MetricKind.validation.value):
        """
        Based on whether the predictions are from the validation or test set, the function populates the
        respective class field with the evaluation metrics.

        Args
            true_labels (ndarray): the true labels of the validation or test set
            predicted_labels (ndarray): the labels predicted by the model
            kind (str): indicates whether we refer to the validation or test set
        """
        if kind == MetricKind.validation.value:
            self.fold_metrics.append(
                {MetricNames.macro_precision.value: precision_score(true_labels, predicted_labels, average="macro"),
                 MetricNames.micro_precision.value: precision_score(true_labels, predicted_labels, average="micro"),
                 MetricNames.macro_recall.value: recall_score(true_labels, predicted_labels, average="macro"),
                 MetricNames.micro_recall.value: recall_score(true_labels, predicted_labels, average="micro"),
                 MetricNames.macro_f.value: f1_score(true_labels, predicted_labels, average="macro"),
                 MetricNames.micro_f.value: f1_score(true_labels, predicted_labels, average="micro")})
        elif kind == MetricKind.test.value:
            self.test_metrics = {
                MetricNames.macro_precision.value: precision_score(true_labels, predicted_labels, average="macro"),
                MetricNames.micro_precision.value: precision_score(true_labels, predicted_labels, average="micro"),
                MetricNames.macro_recall.value: recall_score(true_labels, predicted_labels, average="macro"),
                MetricNames.micro_recall.value: recall_score(true_labels, predicted_labels, average="micro"),
                MetricNames.macro_f.value: f1_score(true_labels, predicted_labels, average="macro"),
                MetricNames.micro_f.value: f1_score(true_labels, predicted_labels, average="micro")}

    def write_fold_results_to_file(self, output_folder, results_folder, fold_num):
        """
        Function to write a fold's results (metrics) to a csv file.

        Args
            output_folder (str): the name of the output folder defined in the configuration yaml file
            results_folder (str): the name of the folder where the results of the current execution will be stored
            fold_num (int): the number of the current fold
        """
        results_folder_path = join(utils.current_dir, output_folder, results_folder)
        if not exists(results_folder_path):
            mkdir(results_folder_path)
        fold_path = join(results_folder_path, "fold_{}".format(fold_num))
        if not exists(fold_path):
            mkdir(fold_path)
        metrics = self.fold_metrics[fold_num]
        df = pd.DataFrame(columns=["classifier", "metric", "result_kind", "result"])
        row = 0
        for metric_name, metric_value in metrics.items():
            df.loc[row] = [self.model_name, metric_name, "validation", metric_value]
            row += 1
        filename = "Results_{}.csv".format(self.model_name)
        file_path = join(fold_path, filename)
        df.to_csv(file_path, sep=',')
        utils.visualize(df, output_folder, results_folder, "fold_{}".format(fold_num),
                        "Plot_fold_{}_{}.png".format(fold_num, self.model_name))

    def write_test_results_to_file(self, output_folder, results_folder):
        """
        Function to write the test results of the model to a csv file.

        Args
            output_folder (str): the name of the output folder defined in the configuration yaml file
            results_folder (str): the name of the folder where the results of the current execution will be stored
        """
        results_folder_path = join(output_folder, results_folder)
        fold_path = join(results_folder_path, "test_results")
        if not exists(fold_path):
            mkdir(fold_path)
        df = pd.DataFrame(columns=["classifier", "metric", "result_kind", "result"])
        row = 0
        for metric_name, metric_value in self.test_metrics.items():
            df.loc[row] = [self.model_name, metric_name, "test", metric_value]
            row += 1
        file_path = join(fold_path, "Results_test_{}.csv".format(self.model_name))
        df.to_csv(file_path, sep=",")
        utils.visualize(df, output_folder, results_folder, "test_results",
                        "Plot_test_{}.png".format(self.model_name))

    def get_fold_avg_result(self, output_folder, results_folder):
        """
        Calculates and writes in a csv file the average value of each metric from all the folds for the specific model.

        Args
            output_folder (str): the name of the output folder defined in the configuration yaml file
            results_folder (str): the name of the folder where the results of the current execution will be stored
        """
        metric_names = [MetricNames.macro_precision.value, MetricNames.micro_precision.value,
                        MetricNames.macro_recall.value, MetricNames.micro_recall.value, MetricNames.macro_f.value,
                        MetricNames.micro_f.value]
        for metric_name in metric_names:
            metric_list = []
            for fold_metric in self.fold_metrics:
                metric_list.append(fold_metric[metric_name])
            self.avg_metrics[metric_name] = sum(metric_list) / len(metric_list)
        results_folder_path = join(output_folder, results_folder)
        avg_folder = join(results_folder_path, "fold_avg")
        if not exists(avg_folder):
            mkdir(avg_folder)
        csv_path = join(avg_folder, "Results_avg.csv")
        df = pd.DataFrame(columns=["classifier", "metric", "result_kind", "result"])
        row = 0
        for metric_name, metric_value in self.avg_metrics.items():
            df.loc[row] = [self.model_name, metric_name, "avg", metric_value]
            row += 1
        df.to_csv(csv_path, sep=",")
        utils.visualize(df, output_folder, results_folder, "fold_avg",
                        "Plot_avg_{}.png".format(self.model_name))

    def find_best_model(self, properties):
        """
        Uses a metric configured by the user to evaluate the best model among the folds.
        The best model is stored in the class field best_model to be later used for testing.

        Args
            properties (dict): properties loaded from the yaml file, uses metric_best_model
        """
        metric_to_compare = properties["metric_best_model"]
        metrics_to_keep = []
        for fold_metric in self.fold_metrics:
            metrics_to_keep.append(fold_metric[metric_to_compare])
        max_value = -1
        max_idx = -1
        for idx, value in enumerate(metrics_to_keep):
            if value > max_value:
                max_value = value
                max_idx = idx
        self.best_model = self.models[max_idx]
        best_models_folder = join(properties["output_folder"], "best_models")
        if not exists(best_models_folder):
            mkdir(best_models_folder)
        best_model_pickle = "best_model_{}_{}.pickle".format(self.model_name, properties["dataset"])
        utils.write_to_pickle(self.best_model, best_models_folder, best_model_pickle)

    def run_cross_validation(self, classifier, properties, input_data, labels, fold_idx, results_folder):
        """
        Checks which classifier is selected and then takes the input data and labels which are divided into k folds.
        Each fold contains a tuple with the train and test indices. For every fold the model is trained and the
        confusion matrix is added to a list.

        Args
            classifier (Classifier): the classifier which is selected
            properties (dict): from properties yaml the value of the key neighbors
            input_data (ndarray): the input data - movie vectors
            labels (ndarray): true labels - ratings
            fold_idx (int): the number of folds
            results_folder (str): the name of the folder where results will be stored
        """

        labels = np.asarray(labels)
        for idx, (train_index, test_index) in enumerate(fold_idx):
            logger.info("Running fold #{}/{}".format(idx + 1, len(fold_idx)))
            input_train, input_test = input_data[train_index], input_data[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            classifier.train(properties=properties, input_data=input_train, labels=labels_train)
            true_labels, predicted_labels = classifier.test(test_data=input_test, true_labels=labels_test)
            self.get_results(true_labels, predicted_labels)
            self.write_fold_results_to_file(properties["output_folder"], results_folder, idx)
