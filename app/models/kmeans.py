from os.path import join, exists
from os import mkdir
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import utils
from utils.enums import CollaborativeModels, Classification, MetricNames, MetricKind
from models.clustering import CollaborativeClustering


class Kmeans(CollaborativeClustering):
    """
    Class that represents KMeans clustering. Uses kmeans form scikit-learn library.
    """

    def __init__(self):
        self.model_name = CollaborativeModels.kmeans.value
        self.metrics = {}
        self.model = None

    def train(self, properties, input_data):
        """
        Runs k-means algorithm and trains the model.

        Args
            properties (dict): kmeans configurations
            input_data (ndarray): vectors with user ratings

        """
        self.model = KMeans(n_clusters=properties["kmeans"]["clusters"], random_state=77, verbose=1,
                            n_init=properties["kmeans"]["n_init"], max_iter=properties["kmeans"]["max_iter"])
        self.model.fit(input_data)

    def test(self, test_data):
        """
        Calculates the distance of every instance to all the clusters.

        Args
            test_data (ndarray): vectors with user ratings

        Returns
            predictions (ndarray): contains the similarity to every cluster for every user
        """
        cluster_distances = self.model.transform(test_data)
        # convert to "similarity" scores
        predictions = 1 - cluster_distances / np.max(cluster_distances)
        return predictions

    def fit_transform(self, properties, input_data):
        """
        Runs the k-means algorithm and obtain the k clusters. Then computes for every instance the distances from every
        cluster and converts these distances into similarities.

        Args
            properties (dict): Configurations of k-means
            input_data (ndarray): The data created in collaborative preprocessing (users ratings)

        Returns
            The lists of similarities between every instance and the clusters
        """
        self.model = KMeans(n_clusters=properties["kmeans"]["clusters"], random_state=77, verbose=1,
                            n_init=properties["kmeans"]["n_init"], max_iter=properties["kmeans"]["max_iter"])
        cluster_distances = self.model.fit_transform(input_data)
        predictions = 1 - cluster_distances / np.max(cluster_distances)
        return predictions

    def calc_results(self, properties, users, logger, classification=Classification.binary.value):
        """
        Calculates the average value of the evaluation metrics of the users' predicted ratings and writes the results
        in a csv file.

        Args
            properties (dict): Output folder, dataset
            users: A list with the predicted ratings for every user
            classification (str): A variable of the class classification
        """
        logger.info("Calculate metrics")
        results_folder = "results_{}_{}".format(self.model_name, properties["dataset"])
        macro_precisions = []
        micro_precisions = []
        macro_recalls = []
        micro_recalls = []
        macro_fs = []
        micro_fs = []
        for idx, user in enumerate(users):
            if type(user.true_rated) == list:
                user.true_rated = np.asarray(user.true_rated)
            if type(user.predicted_rated) == list:
                user.predicted_rated = np.asarray(user.predicted_rated)
            if user.true_rated.size > 0 and user.predicted_rated.size > 0:
                logger.debug("Calculating results for user with id {} and idx {}".format(user.user_id, idx))
                if classification == Classification.binary.value:
                    user.true_rated = np.where(user.true_rated <= 3, 1, 0)
                    user.predicted_rated = np.where(user.predicted_rated <= 3, 1, 0)
                else:
                    user.true_rated = np.around(user.true_rated, decimals=0)
                    user.predicted_rated = np.around(user.predicted_rated, decimals=0)
                macro_precisions.append(precision_score(user.true_rated, user.predicted_rated, average="macro"))
                micro_precisions.append(precision_score(user.true_rated, user.predicted_rated, average="micro"))
                macro_recalls.append(recall_score(user.true_rated, user.predicted_rated, average="macro"))
                micro_recalls.append(recall_score(user.true_rated, user.predicted_rated, average="micro"))
                macro_fs.append(f1_score(user.true_rated, user.predicted_rated, average="macro"))
                micro_fs.append(f1_score(user.true_rated, user.predicted_rated, average="micro"))
        avg_macro_precision = sum(macro_precisions) / len(macro_precisions)
        avg_micro_precision = sum(micro_precisions) / len(micro_precisions)
        avg_macro_recall = sum(macro_recalls) / len(macro_recalls)
        avg_micro_recall = sum(micro_recalls) / len(micro_recalls)
        avg_macro_f = sum(macro_fs) / len(macro_fs)
        avg_micro_f = sum(micro_fs) / len(micro_fs)
        logger.info("Average macro precision: {}".format(avg_macro_precision))
        logger.info("Average micro precision: {}".format(avg_micro_precision))
        logger.info("Average macro recall: {}".format(avg_macro_recall))
        logger.info("Average micro recall: {}".format(avg_micro_recall))
        logger.info("Average macro f: {}".format(avg_macro_f))
        logger.info("Average micro f: {}".format(avg_micro_f))
        df = pd.DataFrame(columns=["classifier", "metric", "result_kind", "result"])
        df.loc[0] = [self.model_name, MetricNames.macro_precision.value, MetricKind.validation.value,
                     avg_macro_precision]
        df.loc[1] = [self.model_name, MetricNames.micro_precision.value, MetricKind.validation.value,
                     avg_micro_precision]
        df.loc[2] = [self.model_name, MetricNames.macro_recall.value, MetricKind.validation.value, avg_macro_recall]
        df.loc[3] = [self.model_name, MetricNames.micro_recall.value, MetricKind.validation.value, avg_micro_recall]
        df.loc[4] = [self.model_name, MetricNames.macro_f.value, MetricKind.validation.value, avg_macro_f]
        df.loc[5] = [self.model_name, MetricNames.micro_f.value, MetricKind.validation.value, avg_micro_f]
        path = join(utils.app_dir, properties["output_folder"], results_folder)
        logger.info("Save metric results to csv")
        filename = "Metric_Results.csv"
        df.to_csv(join(path, filename), sep=",")
        logger.info("Visualize results")
        utils.visualize(df, output_folder=properties["output_folder"], results_folder=results_folder, folder_name=None,
                        filename="Plot_collaborative_{}_{}.png".format(self.model_name, properties["dataset"]))

    def exec_collaborative_method(self, properties, user_ratings, user_ids, movie_ids, logger):
        """
        Calculates the similarity between a target user and the users belonging to the most similar cluster to the
        target user using the pearson correlation coefficient similarity measure. Then it keeps track of the n most
        similar users.

         Args
            properties (dict): kmeans configurations (n_similar)
            user_ratings (ndarray): The users' ratings
            user_ids (ndarray): The users' ids
            movie_ids (ndarray): The movies' ids

        Returns
            A list with the predicted ratings for every user
        """
        if not exists(
                join(utils.app_dir, properties["output_folder"],
                     "results_{}_{}".format(self.model_name, properties["dataset"]))):
            mkdir(join(utils.app_dir, properties["output_folder"],
                       "results_{}_{}".format(self.model_name, properties["dataset"])))
        predictions = self.fit_transform(properties=properties, input_data=user_ratings)
        users = self._find_similar_users(user_ids=user_ids, user_ratings=user_ratings, predictions=predictions)
        for user in users:
            logger.info("Calculating predictions for user with id {}".format(user.user_id))
            user_rating = user_ratings[user.user_idx, :]
            user.average_rating = self.__get_mean_positive_ratings(user_rating)
            logger.info("User with id {} has average rating {}".format(user.user_id, user.average_rating))
            similarities = []
            absolute_similarity_list = []
            logger.info("Calculate pearson similarity with similar users")
            if not user.similar_users:
                continue
            for other_user in user.similar_users:
                logger.debug("Pearson similarity with other user with id {}".format(other_user.user_id))
                other_user_ratings = other_user.user_ratings
                user_same_ratings, other_user_same_ratings, same_movie_ids = self.__find_same_ratings(
                    movie_ids=movie_ids, user_ratings=user_rating, other_user_ratings=other_user_ratings)
                other_user.average_user = self.__get_mean_positive_ratings(other_user.user_ratings)
                if len(same_movie_ids) >= 2:
                    similarity = pearsonr(user_same_ratings, other_user_same_ratings)
                else:
                    similarity = (0, 0)
                logger.debug("User with id {} and user with id {} have similarity {} and in absulute value {}".
                             format(user.user_id, other_user.user_id, similarity[0], abs(similarity[0])))
                similarities.append(similarity[0])
                absolute_similarity_list.append(abs(similarity[0]))
            # sort list from min to max - pearsonr returns p-value
            num_similar_users = properties["kmeans"]["n_similar"]

            similar_users = []
            similarities_final_absolute = []
            similarities_final = []
            logger.info("Find most similar users. Number of similar users {}".format(num_similar_users))
            for i in range(num_similar_users):
                if i < len(similarities):
                    max_idx = absolute_similarity_list.index(max(absolute_similarity_list))
                    similarities_final_absolute.append(absolute_similarity_list[max_idx])
                    similarities_final.append(similarities[max_idx])
                    other_user = user.similar_users[max_idx]
                    logger.debug("Most similar user: {}".format(other_user))
                    logger.debug("Max similarity: {}".format(absolute_similarity_list[max_idx]))
                    similar_users.append(other_user)
                    absolute_similarity_list[max_idx] = -100000000
            logger.info("Calculate user predictions for all movies")
            for movie_idx, movie_id in enumerate(movie_ids):
                user_values = []
                for user_idx, other_user in enumerate(similar_users):
                    similarity = similarities_final[user_idx]
                    avg_other_user = other_user.average_user
                    other_user_rating = other_user.user_ratings[movie_idx]
                    value = (other_user_rating - avg_other_user) * similarity
                    user_values.append(value)
                if sum(similarities_final_absolute) != 0:
                    sum_values = sum(user_values) / sum(similarities_final_absolute)
                else:
                    sum_values = 0
                if user.average_rating + sum_values > 0:
                    user.movie_predictions.append(user.average_rating + sum_values)
                else:
                    user.movie_predictions.append(0.5)
                logger.debug("User prediction for movie id {} is {}".format(movie_id, user.movie_predictions[-1]))
                logger.debug("True rating: {}".format(user.user_ratings[movie_idx]))
            user.user_ratings = np.asarray(user.user_ratings)
            user.movie_predictions = np.asarray(user.movie_predictions)
            indices = list(np.where(user.user_ratings > 0)[0])
            user.true_rated = user.user_ratings[indices]
            user.predicted_rated = user.movie_predictions[indices]
            user.true_rated = np.asarray(user.true_rated)
            user.predicted_rated = np.asarray(user.predicted_rated)
            logger.debug("Save user predictions to csv file")
            self.__write_user_csv(properties, user, movie_ids)
        logger.info("Save all results to pickle file")
        utils.write_to_pickle(users, properties["output_folder"],
                              "collaborative_user_predictions_{}.pickle".format(properties["dataset"]))
        return users

    def __write_user_csv(self, properties, user, movie_ids):
        """
        Writes in a csv file the real ratings and the predicted ratings on movies for a particular user.

        Args
            properties (dict): Output folder, dataset
            user (object): An object of the class user
            movie_ids (ndarray): A list with the movie ids
        """
        path = join(utils.app_dir, properties["output_folder"],
                    "results_{}_{}".format(self.model_name, properties["dataset"]))
        filename = "Predictions_{}.csv".format(user.user_id)
        df = pd.DataFrame(columns=["movie_id", "rating", "prediction"])
        for movie_idx, movie_id in enumerate(movie_ids):
            rating = user.user_ratings[movie_idx]
            prediction = user.movie_predictions[movie_idx]
            df.loc[movie_idx] = [movie_id, rating, prediction]
        file_path = join(path, filename)
        df.to_csv(file_path, sep=',')

    @staticmethod
    def _find_similar_users(user_ids, user_ratings, predictions):
        """
        Sorts the similarities of the predictions list. Then keeps the ratings of every user and the ratings
        of the users belonging to the most similar cluster to the target user.

        Args
            user_ids (ndarray): The users' ids
            user_ratings (ndarray): The ratings of users
            predictions (ndarray): The similarities between the users and the clusters

        Returns
            A list of objects for every user containing the fields of the class user
        """
        users = []
        rows, cols = predictions.shape
        for idx, user_id in enumerate(list(user_ids)):
            user = User(user_id, idx)
            user.user_ratings = user_ratings[idx]
            user_similarities = list(predictions[idx, :])
            max_idx = user_similarities.index(max(user_similarities))
            user.user_cluster_idx = max_idx
            user.similarities = user_similarities
            for row in range(0, rows):
                # check if current and target users are different
                if user.user_id == user_ids[row]:
                    continue
                # checks if the user belongs to the same cluster as the target user
                # get the current user similarities
                other_user_similarities = list(predictions[row, :])
                # find the closest cluster
                other_user_max = other_user_similarities.index(max(other_user_similarities))
                # check if the closest cluster is the same as the target user's closest cluster
                if other_user_max == user.user_cluster_idx:
                    other_user = User(user_ids[row], row)
                    other_user.user_cluster_idx = other_user_max
                    other_user.user_ratings = user_ratings[row]
                    other_user.similarities = other_user_similarities
                    user.similar_users.append(other_user)
            users.append(user)

        return users

    @staticmethod
    def __find_same_ratings(movie_ids, user_ratings, other_user_ratings):
        """
        Find the ratings of two users on the same movies.
        Args
            movie_ids (ndarray): The movie ids
            user_ratings (ndarray): The ratings of the target user
            other_user_ratings (ndarray): The ratings of a similar user to the target user

        Returns
            Three lists one with the ratings of the target and one with the ratings of the similar user and one with
            the ids of movies that both of them rated
        """
        user_same_ratings = []
        other_user_same_ratings = []
        same_movie_ids = []
        for idx, movie_id in enumerate(movie_ids):
            if user_ratings[idx] > 0 and other_user_ratings[idx] > 0:
                user_same_ratings.append(user_ratings[idx])
                other_user_same_ratings.append(other_user_ratings[idx])
                same_movie_ids.append(movie_id)
        return user_same_ratings, other_user_same_ratings, same_movie_ids

    @staticmethod
    def __get_mean_positive_ratings(ratings):
        """
        From a list of ratings find those with positive value and calculate their mean.

        Args
            ratings (ndarray): A list of ratings

        Returns
             The mean of the positive ratings
        """
        positives = ratings > 0
        if positives.any():
            return ratings[positives].mean()


class User:
    """
    Class that represents users.
    """
    user_id = -1
    user_idx = -1
    similar_users = []
    user_ratings = []
    true_rated = []
    similarities = []
    user_cluster_idx = -1
    average_rating = 0.0
    movie_predictions = []
    predicted_rated = []

    def __init__(self, user_id, user_idx):
        self.user_id = user_id
        self.user_idx = user_idx
        self.similar_users = []
        self.user_ratings = []
        self.user_cluster_idx = -1
