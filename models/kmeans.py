import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans

from enums import CollaborativeModels
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

    def exec_collaborative_method(self, properties, user_ratings, user_ids, movie_ids):
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
        predictions = self.fit_transform(properties=properties, input_data=user_ratings)
        users = self._find_similar_users(user_ids=user_ids, user_ratings=user_ratings, predictions=predictions)
        for user in users:
            user_rating = user_ratings[user.user_idx, :]
            similarities = []
            for other_user in user.similar_users:
                other_user_ratings = other_user.user_ratings
                similarity = pearsonr(user_rating, other_user_ratings)
                similarities.append(similarity)
            # sort list from min to max - pearsonr returns p-value
            num_similar_users = properties["kmeans"]["n_similar"]
            similar_users = []
            for i in range(num_similar_users):
                if i < len(similarities):
                    min_idx = similarities.index(min(similarities))
                    other_user = user.similar_users[min_idx]
                    similar_users.append(other_user)
                    similarities[min_idx] = 100000000
            # TODO predict ratings

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
        for row in range(0, rows):
            predictions[row, :].sort()
        for idx, user_id in enumerate(np.array(user_ids).tolist()):
            user = User(user_id, idx)
            user.user_ratings = user_ratings[idx]
            user.user_similarities = predictions[idx, 0]
            for row in range(0, rows):
                # checks if the user belongs to the same cluster as the target user
                if predictions[row, 0] == user.user_similarities and user.user_id != user_ids[row]:
                    other_user = User(user_ids[row], row)
                    other_user.user_ratings = user_ratings[row]
                    user.similar_users.append(other_user)
            users.append(user)
        return users


class User:
    """
    Class that represents users.
    """
    user_id = -1
    user_idx = -1
    similar_users = []
    user_ratings = []
    user_similarities = []

    def __init__(self, user_id, user_idx):
        self.user_id = user_id
        self.user_idx = user_idx
        self.similar_users = []
        self.user_ratings = []
        self.user_similarities = []
