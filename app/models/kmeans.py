import numpy as np
from sklearn.cluster import KMeans

from models.models import User
from utils.enums import CollaborativeModels


class Kmeans:
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
    def find_similar_users(user_ids, user_ratings, predictions):
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
            user_similarities = predictions[idx, :].tolist()
            max_idx = np.argmax(user_similarities)
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
