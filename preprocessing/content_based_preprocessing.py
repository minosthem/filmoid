import os
import re
import string

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import utils
from enums import PreprocessKind, Classification, AggregationStrategy
from preprocessing.data_preprocessing import DataPreprocessing
from utils import logger


class ContentBasedPreprocessing(DataPreprocessing):
    """
    Class that extends the DataPreprocessing class. Used for the content based method in recommendation systems.
    Overrides the preprocess method to generate the input vectors, i.e. a vector containing user and movie information.
    For the users, the only information is their id, so the first dimension of the input vectors is the user id. For
    the movies, the information used is the title, the genres and the tags given by the specific user. All those text
    are concatenated into a single string and its words are replaced by word embeddings. As labels, the specific rating
    is used (0 or 1 for binary classification - 5 classes for multi-class classification).
    """
    input_data_pickle = "input_data.pickle"
    ratings_pickle = "ratings.pickle"
    test_dataset_pickle = "test_recommendation.pickle"
    punct_digit_to_space = str.maketrans(string.punctuation + string.digits,
                                         " " * len(string.punctuation + string.digits))

    input_data = None
    ratings = None

    def preprocess(self, properties, datasets, kind=PreprocessKind.train.value):
        """
            Checks if the input and the rating file exist and loads them from the output folder. Otherwise, takes the
            rating, movies and tags datasets and converts them to dataframes and also loads the glove file. It iterates
            the ratings dataframe keeping from every row the movie id, user id and the rating. It uses the functions
            preprocess_rating, preprocess_text and text_to_glove to create a vector corresponding to a movie's features
            and user id. The user's id is added on the first position of that vector. Every vector is added to a list of
            vectors called input_data. Finally, the rating of every user for a particular movie is added to a list
            called ratings and both this list as well as the input_data list are being saved to the output folder.

            Args:
                properties(dict): properties loaded from yaml file. Used so as to get the output folder
                datasets (dict): contains the dataframes of all the movielens csvs
                kind (str): if set to train the ratings.csv is used for input vectors otherwise the generated
                test_recommendation.csv is used
        """
        output_folder = properties["output_folder"]
        input_data_pickle_filename = self.input_data_pickle + "_{}_{}".format(properties["dataset"],
                                                                              properties["classification"])
        ratings_pickle_filename = self.ratings_pickle + "_{}_{}".format(properties["dataset"],
                                                                        properties["classification"])
        test_dataset_pickle_filename = self.test_dataset_pickle + "_{}_{}".format(properties["dataset"],
                                                                                  properties["classification"])

        if utils.check_file_exists(output_folder, input_data_pickle_filename) and \
                utils.check_file_exists(output_folder, ratings_pickle_filename):
            logger.info("Content-based input data already exist and will be loaded from pickle file")
            input_filename = input_data_pickle_filename if kind == PreprocessKind.train.value else \
                test_dataset_pickle_filename
            self.input_data = utils.load_from_pickle(output_folder, input_filename)
            self.ratings = utils.load_from_pickle(output_folder, ratings_pickle_filename)
            logger.info("Loaded inputs of shape {}".format(self.input_data.shape))
            logger.info("Loaded ratings of shape {}".format(self.ratings.shape))
        else:
            os.makedirs(output_folder, exist_ok=True)
            ratings_df = datasets["ratings"] if kind == PreprocessKind.train.value else datasets["test_recommendation"]
            movies_df = datasets["movies"]
            tags_df = datasets["tags"]
            glove_df = utils.load_glove_file(properties)
            logger.info("Generating input vectors")
            self.input_data = []
            self.ratings = []
            for index, row in ratings_df.iterrows():
                user_id, movie_id, rating, _ = row
                movie_id = int(movie_id)
                user_id = int(user_id)
                # preprocess
                rating = self._preprocess_rating(properties, rating)
                movie_text = self._preprocess_text(movies_df, tags_df, movie_id, user_id)

                movie_vector = self._text_to_glove(properties, glove_df, movie_text)
                if movie_vector.size == 0:
                    continue
                movie_vector = np.insert(movie_vector, 0, user_id, axis=1)
                self.input_data.append(movie_vector)
                self.ratings.append(rating)
                utils.print_progress(self.ratings)

            self.ratings = np.asarray(self.ratings)
            self.input_data = np.concatenate(self.input_data)
            logger.info("Produced a feature matrix of shape {}".format(self.input_data.shape))
            # standardization
            logger.info("Standardize input vectors")
            self.input_data = preprocessing.scale(self.input_data)
            logger.info("Save input vectors to file")
            input_filename = input_data_pickle_filename if kind == PreprocessKind.train.value else \
                test_dataset_pickle_filename
            utils.write_to_pickle(obj=self.input_data, directory=output_folder, filename=input_filename)
            utils.write_to_pickle(obj=self.ratings, directory=output_folder, filename=ratings_pickle_filename)

    @staticmethod
    def _text_to_glove(properties, glove_df, word_list):
        """
        Takes the pre-processed text created by the preprocess_text function and converts it to a vector of numbers
        using the word embeddings from Glove (https://nlp.stanford.edu/projects/glove/).
        This process is being done for every word of the text separately. In the end all the vectors
        are gathered in a list of vectors (embeddings) and, depending on the aggregation policy defined in the
        properties yaml file, it creates a single vector containing the average or the maximum number of every position
        of all the word embeddings vectors.

        Args
            properties (dict): loaded properties from yaml, uses the aggregation strategy (avg or max)
            glove_df (DataFrame): the file with the Glove embeddings
            word_list (list): movie text split into its words as list

        Returns
            ndarray: a vector for every movie text which contains the movie title and genre and also the tags of a user
            for a specific movie
        """
        embeddings = []
        for word in word_list:
            # to lowercase
            word = word.lower()
            if word not in glove_df.index:
                continue
            # expand dimensions to a 1 x <dim> vector
            embeddings.append(np.expand_dims(glove_df.loc[word].values, 0))

        # concatenate to a <num words> x <dimension> matrix
        embeddings = np.concatenate(embeddings, axis=0)

        # aggregate all word vectors to a single vector
        if properties["aggregation"] == AggregationStrategy.average.value:
            embeddings = np.mean(embeddings, axis=0)
        elif properties["aggregation"] == AggregationStrategy.max.value:
            embeddings = np.max(embeddings, axis=0)

        # expand dimension to a 1 x <dim> vector
        return np.expand_dims(embeddings, 0)

    def _preprocess_text(self, movies_df, tags_df, movie_id, user_id):
        """
        It keeps the movie id from the movies dataset and also the tags of a particular user for the specific movie.
        Then, it creates a string containing the movie title and movie genre which are taken from the movies dataset as
        well as the. Finally, it adds the tags of the user for this movie to the string and gets rid of every symbol
        and number of that text.

        Args
            movies_df (DataFrame): movies dataset
            tags_df (DataFrame): tags dataset
            movie_id (int): id of a movie
            user_id (int): id of a user

        Returns
            list: the preprocessed text containing the title, genre and tags for the movie split into words
        """
        m = movies_df[movies_df["movieId"] == movie_id]
        tags = tags_df[(tags_df["userId"] == user_id) & (tags_df["movieId"] == movie_id)]
        movie_title = m.iloc[0]["title"]
        movie_genres = m.iloc[0]["genres"]
        movie_text = movie_title + " " + movie_genres
        if not tags.empty:
            tags = tags["tag"]
            for index, row in tags.iteritems():
                movie_text = movie_text + " " + row
        # preprocessing title, genres, tags ==> remove symbols, numbers
        # remove digits and punctuation
        movie_text = movie_text.translate(self.punct_digit_to_space).strip()
        # merge multiple spaces
        movie_text = re.sub("[ ]+", " ", movie_text)
        return movie_text.split()

    @staticmethod
    def _preprocess_rating(properties, rating):
        """
        Converts a rating to a binary score o and 1 if the classification policy is binary else it keeps the rating and
        rounds it (uses the ratings as Integer numbers).

        Args
            properties (dict): loaded from yaml file, uses the classification parameter
            rating (float): the rating of a user for a specific movie

        Returns
            int: 0 or 1 for binary classification, 1-5 numbers for multi-class classification
        """
        if properties["classification"] == Classification.binary.value:
            return 0 if rating > 3 else 1
        else:
            return round(rating)

    def create_train_test_data(self, input_data, labels):
        """
        It splits the input data and the labels into a train dataset with the corresponding labels and a test dataset
        with the corresponding labels.

        Args
            input_data (ndarray): the input vectors of the ratings dataset
            labels (ndarray): the labels of the input_data

        Returns
            ndarray: the train and test dataset and labels
        """
        input_train, input_test, labels_train, labels_test = train_test_split(input_data, labels,
                                                                              test_size=0.2,
                                                                              random_state=0)
        return input_train, input_test, labels_train, labels_test

    def create_cross_validation_data(self, input_data, properties):
        """
        Takes the input data and creates k folds depending on the number of folds mentioned in the properties file.

        Args
            input_data (ndarray): the training set to be used for k-fold cross-validation
            properties (dict): loaded from yaml file, uses the cross-validation parameter to define the number of folds

        Returns
            FoldGenerator: the generator of the k-fold indices
        """
        kf = KFold(n_splits=properties["cross-validation"], shuffle=True, random_state=666)
        return kf.split(input_data)
