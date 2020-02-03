import numpy as np
from sklearn.cluster import KMeans

from models.clustering import CollaborativeClustering


class Kmeans(CollaborativeClustering):
    """
    Class that represents KMeans clustering. Uses kmeans form scikit-learn library.
    """

    def __init__(self):
        self.model_name = "kmeans"
        self.metrics = {}
        self.model = None

    def train(self, properties, input_data):
        self.model = KMeans(n_clusters=properties["kmeans"]["clusters"], random_state=77, verbose=1,
                            n_init=properties["kmeans"]["n_init"], max_iter=properties["kmeans"]["max_iter"])
        self.model.fit(input_data)

    def test(self, test_data):
        cluster_distances = self.model.transform(test_data)
        # convert to "similarity" scores
        predictions = 1 - cluster_distances / np.max(cluster_distances)
        return predictions

    def fit_transform(self, properties, input_data):
        self.model = KMeans(n_clusters=properties["kmeans"]["clusters"], random_state=77, verbose=1,
                            n_init=properties["kmeans"]["n_init"], max_iter=properties["kmeans"]["max_iter"])
        cluster_distances = self.model.fit_transform(input_data)
        predictions = 1 - cluster_distances / np.max(cluster_distances)
        return predictions
