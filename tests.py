from os.path import join
from random import randint
import unittest
import os
import pandas as pd
import numpy as np
import utils

from preprocessing import data_preprocessing as dp
from models import classifiers, results


class TestUtilMethods(unittest.TestCase):

    def test_load_properties(self):
        utils.properties_file = join(os.getcwd(), "properties", "properties.yaml")
        properties = utils.load_properties()
        self.assertEqual(properties["datasets_folder"], "Datasets")

    def test_check_file_exists(self):
        file_exist = utils.check_file_exists("resources", "glove.6B.50d.txt")
        self.assertTrue(file_exist)

    def test_get_filenames(self):
        properties = {"datasets_folder": "Datasets", "dataset": "ml-dev",
                      "filenames": ["links", "movies", "ratings", "tags"], "dataset-file-extention": ".csv"}
        filenames = utils.get_filenames(properties)
        self.assertEqual(len(filenames), 4)

    def test_load_glove_file(self):
        properties = {"resources_folder": "resources", "embeddings_file": "glove.6B.50d.txt"}
        df = utils.load_glove_file(properties)
        self.assertTrue(not df.empty)


class TestDataPreProcessing(unittest.TestCase):

    def test_read_csv(self):
        properties = {"datasets_folder": "Datasets", "dataset": "ml-dev",
                      "filenames": ["links", "movies", "ratings", "tags"], "dataset-file-extention": ".csv"}
        files = {}
        folder_path = join(os.getcwd(), properties["datasets_folder"], properties["dataset"])
        for file in properties["filenames"]:
            filename = file + properties["dataset-file-extention"]
            files[file] = join(folder_path, filename)
        datasets = dp.read_csv(files)
        self.assertEqual(len(datasets.keys()), 4)
        for dataset, df in datasets.items():
            self.assertTrue(not df.empty)

    def test_create_train_test_data(self):
        input_data, labels = np.arange(10).reshape((5, 2)), range(5)
        input_train, input_test, labels_train, labels_test = dp.create_train_test_data(input_data=input_data,
                                                                                       labels=labels)
        self.assertEqual(input_train.shape, (4, 2))
        self.assertEqual(input_test.shape, (1, 2))
        self.assertEqual(len(labels_train), 4)
        self.assertEqual(len(labels_test), 1)

    def test_create_cross_validation_data(self):
        properties = {"cross-validation": 2}
        input_data = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        folds = dp.create_cross_validation_data(input_data=input_data, properties=properties)
        count = 0
        for idx, (train_index, test_index) in enumerate(folds):
            self.assertEqual(train_index.shape, (2,))
            self.assertEqual(test_index.shape, (2,))
            count += 1
        self.assertEqual(2, count)

    def test_text_to_glove(self):
        word_list = ["Toy", "Story", "Adventure", "Animation", "Children", "Comedy", "Fantasy", "funny"]
        data = [["toy", 1, 1, 1, 1, 1],
                ["story", 2, 2, 2, 2, 2],
                ["adventure", 3, 3, 3, 3, 3],
                ["animation", 4, 4, 4, 4, 4],
                ["children", 5, 5, 5, 5, 5],
                ["comedy", 6, 6, 6, 6, 6]]
        glove_df = pd.DataFrame(data=data, columns=None)
        glove_df = glove_df.set_index(0)
        properties = {"aggregation": "avg"}
        expected_vector = np.array([[3.5, 3.5, 3.5, 3.5, 3.5]])
        text_vector = dp.text_to_glove(properties=properties, glove_df=glove_df, word_list=word_list)
        self.assertEqual(text_vector.all(), expected_vector.all())

    def test_preprocess_text(self):
        movies_df = pd.DataFrame(data=[[1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"]],
                                 columns=["movieId", "title", "genres"])
        tags_df = pd.DataFrame(data=[[1, 1, "funny"]], columns=["userId", "movieId", "tag"])
        movie_id = 1
        user_id = 1
        text = dp.preprocess_text(movies_df=movies_df, tags_df=tags_df, movie_id=movie_id, user_id=user_id)
        expected_text = ["Toy", "Story", "Adventure", "Animation", "Children", "Comedy", "Fantasy", "funny"]
        self.assertEqual(text, expected_text)

    def test_preprocess_rating(self):
        # test cases for binary classification
        properties = {"classification": "binary"}
        # case dislike
        rating = 1.5
        expected_rating = 1
        new_rating = dp.preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # case like
        rating = 4
        expected_rating = 0
        new_rating = dp.preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # case dislike
        rating = 2.9
        expected_rating = 1
        new_rating = dp.preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)

        # test rating for multi-class classification
        properties["classification"] = "multi"
        # round rating
        rating = 1.5
        expected_rating = 2
        new_rating = dp.preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # rating remains the same
        rating = 3
        expected_rating = 3
        new_rating = dp.preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # round rating
        rating = 4.55
        expected_rating = 5
        new_rating = dp.preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)


class TestClassifiers(unittest.TestCase):

    def test_knn_classifier(self):
        knn = classifiers.KNN()
        matrices = self.run_classifier(knn)
        self.assertEqual(len(matrices), 2)
        self.assertTrue(type(knn) == classifiers.KNN)

    def test_random_forest(self):
        rf = classifiers.RandomForest()
        matrices = self.run_classifier(rf)
        self.assertEqual(len(matrices), 2)
        self.assertTrue(type(rf) == classifiers.RandomForest)

    @staticmethod
    def run_classifier(classifier):
        properties = {"knn": {"neighbors": 5}, "rf": {"estimators": 100,
                                                      "max_depth": 10}, "cross-validation": 2}
        input_data, labels = np.arange(1000).reshape((100, 10)), [randint(1, 5) for _ in range(100)]
        input_train, input_test, labels_training, labels_testing = dp.create_train_test_data(input_data=input_data,
                                                                                             labels=labels)
        labels_training = np.asarray(labels_training)
        folds = dp.create_cross_validation_data(input_data=input_train, properties=properties)
        matrices = []
        fold_idx = list(folds)
        for idx, (train_idx, test_idx) in enumerate(fold_idx):
            print("Running fold #{}/{}".format(idx + 1, len(fold_idx)))
            input_training, input_testing = input_train[train_idx], input_train[test_idx]
            labels_train, labels_test = labels_training[train_idx], labels_training[test_idx]
            classifier.train(input_training, labels_train, properties)
            conf_matrix = classifier.test(input_testing, labels_test)
            matrices.append(conf_matrix)
        return matrices


class TestResults(unittest.TestCase):

    def test_calc_results(self):
        test_classifiers = TestClassifiers()
        knn = classifiers.KNN()
        matrices = test_classifiers.run_classifier(knn)
        matrix = matrices[0]
        properties = {"classification": "multi"}
        metrics = results.calc_results(properties=properties, confusion_matrix=matrix)
        self.assertEqual(len(metrics), 6)
