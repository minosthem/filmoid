class Clustering:
    """
    Generic class that models clustering methods. Clustering is used in the collaborative method of recommendation
    systems.
    """

    def train(self, properties, input_data):
        pass

    def test(self, test_data):
        pass


class CollaborativeClustering(Clustering):
    """
    Class to be extended by collaborative models (e.g. kmeans). Methods train and test should be implemented by the
    sub-classes.
    """

    metrics = {}
    model_name = ""

    def train(self, properties, input_data):
        pass

    def test(self, test_data):
        pass

    def fit_transform(self, properties, input_data):
        pass

    def exec_collaborative_method(self, properties, user_ratings, user_ids, movie_ids):
        pass
