from os.path import join
from random import randint
import unittest
import os
import pandas as pd
import numpy as np
import utils

from preprocessing.data_preprocessing import DataPreprocessing
from preprocessing.content_based_preprocessing import ContentBasedPreprocessing
from results import results
from models.knn_classifier import KNN
from models.rf_classifier import RandomForest


class TestUtilMethods(unittest.TestCase):
    """
    Unit test class to evaluate all the utility methods from utils.py
    """

    def test_load_properties(self):
        """
        Tests whether the properties yaml is loaded

        Examined test case: the value for the datasets_folder
        """
        utils.properties_file = join(os.getcwd(), "properties", "properties.yaml")
        properties = utils.load_properties()
        self.assertEqual(properties["datasets_folder"], "Datasets")

    def test_check_file_exists(self):
        """
        Test case for check_file_exists method in utils.py

        Examined test case: smallest glove file exists in resources folder
        """
        file_exist = utils.check_file_exists("resources", "glove.6B.50d.txt")
        self.assertTrue(file_exist)

    def test_get_filenames(self):
        """
        Tests the get_filenames method for the links, movies, ratings and tags csv files. The purpose is to create
        a dictionary with keys the name of the files (without the file extension) and values the full path to those
        files.

        Examined test case: dictionary contains 4 key-value pairs
        """
        properties = {"datasets_folder": "Datasets", "dataset": "ml-dev",
                      "filenames": ["links", "movies", "ratings", "tags"], "dataset-file-extention": ".csv"}
        filenames = utils.get_filenames(properties)
        self.assertEqual(len(filenames), 4)

    def test_load_glove_file(self):
        """
        Test method to load the glove file as DataFrame

        Examined test case: just checks that the DataFrame is not empty
        """
        properties = {"resources_folder": "resources", "embeddings_file": "glove.6B.50d.txt"}
        df = utils.load_glove_file(properties)
        self.assertTrue(not df.empty)


class TestDataPreProcessing(unittest.TestCase):
    """
    Class for the data_preprocessing.py file
    """

    def test_read_csv(self):
        """
        Method to test the read_csv function. Given a dictionary containing the full path to the csv files to be read,
        the function should return a dictionary with keys the name of the files (without the file extension) and values
        the DataFrames.

        Examined test cases:

        1. The returned dictionary contains 4 key-value pairs
        2. No dataframe is empty
        """
        properties = {"datasets_folder": "Datasets", "dataset": "ml-dev",
                      "filenames": ["links", "movies", "ratings", "tags"], "dataset-file-extention": ".csv"}
        files = {}
        folder_path = join(os.getcwd(), properties["datasets_folder"], properties["dataset"])
        for file in properties["filenames"]:
            filename = file + properties["dataset-file-extention"]
            files[file] = join(folder_path, filename)
        data_preprocess = DataPreprocessing()
        data_preprocess.read_csv(files)
        datasets = data_preprocess.datasets
        self.assertEqual(len(datasets.keys()), 4)
        for dataset, df in datasets.items():
            self.assertTrue(not df.empty)

    def test_create_train_test_data(self):
        """
        Test method for the create_train_test_data function. The purpose of the function is to split a given dataset
        into training and test datasets by keeping 20% of the data as test.

        Examined test cases:

        1. Size of input train data
        2. Size of input test data
        3. Length of training labels list
        4. Length of testing labels list
        """
        input_data, labels = np.arange(10).reshape((5, 2)), range(5)
        dp = ContentBasedPreprocessing()
        input_train, input_test, labels_train, labels_test = dp.create_train_test_data(input_data=input_data,
                                                                                       labels=labels)
        self.assertEqual(input_train.shape, (4, 2))
        self.assertEqual(input_test.shape, (1, 2))
        self.assertEqual(len(labels_train), 4)
        self.assertEqual(len(labels_test), 1)

    def test_create_cross_validation_data(self):
        """
        Test method for the create_cross_validation_data. The training dataset is used to generate k folds.

        Examined test cases:

        1. The size of train indices
        2. The size of test indices
        3. The number of the generated folds
        """
        properties = {"cross-validation": 2}
        input_data = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        dp = ContentBasedPreprocessing()
        folds = dp.create_cross_validation_data(input_data=input_data, properties=properties)
        count = 0
        for idx, (train_index, test_index) in enumerate(folds):
            self.assertEqual(train_index.shape, (2,))
            self.assertEqual(test_index.shape, (2,))
            count += 1
        self.assertEqual(2, count)

    def test_text_to_glove(self):
        """
        Method to test the functionality of the text_to_glove function. Given a list of words and a DataFrame of
        word embeddings (words represented as vectors) the method transforms the list into list of vectors following
        an aggregation strategy (avg or max).

        Examined test case: given and expected input vectors are the same
        """
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
        data_preprocess = ContentBasedPreprocessing()
        text_vector = data_preprocess._text_to_glove(properties=properties, glove_df=glove_df, word_list=word_list)
        self.assertEqual(text_vector.all(), expected_vector.all())

    def test_preprocess_text(self):
        """
        Test method for the preprocess_text function. Given a movie and user id, the movie title, genres and given tags
        by the user are collected and concatenated into a single text. Then the text is preprocessed by removing symbols
        and numbers and splitting the text into a list of words.

        Examined test case: the returned list of words is the same as the expected list of words
        """
        movies_df = pd.DataFrame(data=[[1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"]],
                                 columns=["movieId", "title", "genres"])
        tags_df = pd.DataFrame(data=[[1, 1, "funny"]], columns=["userId", "movieId", "tag"])
        movie_id = 1
        user_id = 1
        data_preprocess = ContentBasedPreprocessing()
        text = data_preprocess._preprocess_text(movies_df=movies_df, tags_df=tags_df, movie_id=movie_id,
                                                user_id=user_id)
        expected_text = ["Toy", "Story", "Adventure", "Animation", "Children", "Comedy", "Fantasy", "funny"]
        self.assertEqual(text, expected_text)

    def test_preprocess_rating(self):
        """
        Test method for preprocess_rating function. Based on the classification (binary or multi-class) the rating
        values are replaced by 0,1 or 1,2,3,4,5 respectively.

        Examined test cases:

            1. Binary classification
                a. rating for dislike
                b. rating for like
            2. Multi-class classification
                a. round rating to the next number
                b. same rating
                c. round rating to the same number without the decimal numbers
        """
        data_preprocessing = ContentBasedPreprocessing()
        # test cases for binary classification
        properties = {"classification": "binary"}
        # case dislike
        rating = 1.5
        expected_rating = 1
        new_rating = data_preprocessing._preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # case like
        rating = 4
        expected_rating = 0
        new_rating = data_preprocessing._preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # case dislike
        rating = 2.9
        expected_rating = 1
        new_rating = data_preprocessing._preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)

        # test rating for multi-class classification
        properties["classification"] = "multi"
        # round rating
        rating = 1.5
        expected_rating = 2
        new_rating = data_preprocessing._preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # rating remains the same
        rating = 3
        expected_rating = 3
        new_rating = data_preprocessing._preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)
        # round rating
        rating = 4.22
        expected_rating = 4
        new_rating = data_preprocessing._preprocess_rating(properties, rating)
        self.assertEqual(new_rating, expected_rating)


class TestClassifiers(unittest.TestCase):
    """
    Class to test the classifiers.py file
    """

    def test_knn_classifier(self):
        """
        Test function for knn classifier

        Examined test cases:

        1. Number of confusion matrices returned based on the requested folds
        2. Type of the classifier is equal to classifiers.KNN
        """
        knn = KNN()
        matrices = self.run_classifier(knn)
        self.assertEqual(len(matrices), 2)
        self.assertTrue(type(knn) == KNN)

    def test_random_forest(self):
        """
        Test function for Random Forest classifier

        Examined test cases:

        1. Number of confusion matrices returned based on the requested folds
        2. Type of the classifier is equal to classifiers.RandomForest
        """
        rf = RandomForest()
        matrices = self.run_classifier(rf)
        self.assertEqual(len(matrices), 2)
        self.assertTrue(type(rf) == RandomForest)

    @staticmethod
    def run_classifier(classifier):
        """
        Method used by the classifiers testing methods to implement the training and the testing of the classifier.
        :param classifier: the model to be tested
        :return: the confusion matrices of all the folds
        """
        properties = {"knn": {"neighbors": 5}, "rf": {"estimators": 100,
                                                      "max_depth": 10}, "cross-validation": 2}
        input_data, labels = np.arange(1000).reshape((100, 10)), [randint(1, 5) for _ in range(100)]
        dp = ContentBasedPreprocessing()
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
            classifier.train(properties, input_training, labels_train)
            conf_matrix = classifier.test(input_testing, labels_test)
            matrices.append(conf_matrix)
        return matrices


class TestResults(unittest.TestCase):
    """
    Class to test the functionality of the methods in the results.py
    """

    def test_calc_results(self):
        """
        Testing the method calc_results for the KNN classifier. The method should calculate the micro/macro precision,
        recall and F-measure

        Examined test case: the number of the returned metrics for the 1st fold is equal to 6
        """
        test_classifiers = TestClassifiers()
        knn = KNN()
        matrices = test_classifiers.run_classifier(knn)
        matrix = matrices[0]
        properties = {"classification": "multi"}
        metrics = results.calc_results(properties=properties, confusion_matrix=matrix)
        self.assertEqual(len(metrics), 6)
