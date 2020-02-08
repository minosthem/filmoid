from os.path import join, exists
from os import mkdir
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import utils
from utils.enums import CollaborativeModels, Classification, MetricNames
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

    @staticmethod
    def calc_results(properties, users, classification=Classification.binary.value):
        macro_precisions = []
        micro_precisions = []
        macro_recalls = []
        micro_recalls = []
        macro_fs = []
        micro_fs = []
        for user in users:
            if classification == Classification.binary.value:
                user.true_rated[user.true_rated <= 3] = 1
                user.true_rated[user.true_rated > 3] = 0
                user.predicted_rated[user.predicted_rated <= 3] = 1
                user.predicted_rated[user.predicted_rated > 3] = 0
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
        avg_micro_f = sum(macro_fs) / len(micro_fs)
        df = pd.DataFrame(columns=["classifier", "metric", "result_kind", "result"])
        df.loc[0] = ["kmeans", MetricNames.macro_precision.value, "validation", avg_macro_precision]
        df.loc[1] = ["kmeans", MetricNames.micro_precision.value, "validation", avg_micro_precision]
        df.loc[2] = ["kmeans", MetricNames.macro_recall.value, "validation", avg_macro_recall]
        df.loc[3] = ["kmeans", MetricNames.micro_recall.value, "validation", avg_micro_recall]
        df.loc[4] = ["kmeans", MetricNames.macro_f.value, "validation", avg_macro_f]
        df.loc[5] = ["kmeans", MetricNames.macro_f.value, "validation", avg_micro_f]
        path = join(utils.app_dir, properties["output_folder"], "results_kmeans_{}".format(properties["dataset"]))
        filename = "Metric_Results.csv"
        df.to_csv(join(path, filename), sep=",")

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
                join(utils.app_dir, properties["output_folder"], "results_kmeans_{}".format(properties["dataset"]))):
            mkdir(join(utils.app_dir, properties["output_folder"], "results_kmeans_{}".format(properties["dataset"])))
        predictions = self.fit_transform(properties=properties, input_data=user_ratings)
        users = self._find_similar_users(user_ids=user_ids, user_ratings=user_ratings, predictions=predictions)
        for user in users:
            logger.info("Calculating predictions for user with id {}".format(user.user_id))
            user_rating = user_ratings[user.user_idx, :]
            user.average_rating = self.__get_mean_positive_ratings(user_rating)
            similarities = []
            logger.info("Calculate pearson similarity with similar users")
            for other_user in user.similar_users:
                logger.debug("Pearson similarity with other user with id {}".format(other_user.user_id))
                other_user_ratings = other_user.user_ratings
                user_same_ratings, other_user_same_ratings, same_movie_ids = self.__find_same_ratings(
                    movie_ids=movie_ids, user_ratings=user_rating, other_user_ratings=other_user_ratings)
                other_user.average_user = self.__get_mean_positive_ratings(other_user.user_ratings)
                similarity = pearsonr(user_same_ratings, other_user_same_ratings)
                if similarity[0] < 0:
                    continue
                similarities.append(similarity[0])
            # sort list from min to max - pearsonr returns p-value
            num_similar_users = properties["kmeans"]["n_similar"]
            similar_users = []
            similarities_final = []
            for i in range(num_similar_users):
                if i < len(similarities):
                    max_idx = similarities.index(max(similarities))
                    similarities_final.append(similarities[max_idx])
                    other_user = user.similar_users[max_idx]
                    similar_users.append(other_user)
                    similarities[max_idx] = -100000000
            for movie_idx, movie_id in enumerate(movie_ids):
                user_values = []
                for user_idx, other_user in enumerate(similar_users):
                    similarity = similarities_final[user_idx]
                    avg_other_user = other_user.average_user
                    other_user_rating = other_user.user_ratings[movie_idx]
                    value = (other_user_rating - avg_other_user) * similarity
                    user_values.append(value)
                sum_values = sum(user_values) / abs(sum(similarities_final))
                user.movie_predictions.append(user.average_rating + sum_values)
            user.user_ratings = np.asarray(user.user_ratings)
            user.movie_predictions = np.asarray(user.movie_predictions)
            indices = user.user_ratings > 0
            user.true_rated = user.user_ratings[indices]
            user.predicted_rated = user.movie_predictions[indices]
            self.__write_user_csv(properties, user, movie_ids)
        utils.write_to_pickle(users, properties["output_folder"],
                              "collaborative_user_predictions_{}.pickle".format(properties["dataset"]))
        return users

    @staticmethod
    def __write_user_csv(properties, user, movie_ids):
        path = join(utils.app_dir, properties["output_folder"], "results_kmeans_{}".format(properties["dataset"]))
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
