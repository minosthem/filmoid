import unittest
import pandas as pd
import utils

from preprocessing import data_preprocessing as dp
from models import classifiers, results


class TestUtilMethods(unittest.TestCase):

    def test_load_properties(self):
        pass

    def test_check_file_exists(self):
        pass

    def test_get_filenames(self):
        pass

    def test_load_glove_file(self):
        pass


class TestDataPreProcessing(unittest.TestCase):

    def test_read_csv(self):
        pass

    def test_create_train_test_data(self):
        pass

    def test_create_cross_validation_data(self):
        pass

    def test_text_to_glove(self):
        pass

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
        properties = utils.load_properties()
        # test cases for binary classification
        properties["classification"] = "binary"
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


class TestResults(unittest.TestCase):

    def test_calc_results(self):
        results.calc_results()
