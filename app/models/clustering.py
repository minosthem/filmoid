from utils.enums import Classification


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
        """
        Method to be implemented by the children classes. Used to train a model.

        Args
            properties (dict): the loaded configuration file
            input_data (ndarray): the input vectors for the training

        Returns
            NotImplementedError: raises an exception if the child class has not implemented the method
        """
        raise NotImplementedError

    def test(self, test_data):
        """
        Method to be implemented by the children classes. Used to test a model.

        Args
            test_data (ndarray): the test set

        Returns
            NotImplementedError: raises an exception if the method is not implemented by the children classes
        """
        raise NotImplementedError

    def fit_transform(self, properties, input_data):
        """
        Method should be implemented by the children classes. Used to both train and test a model with the dataset.

        Args
            properties (dict): the loaded configuration file
            input_data (ndarray): the input vectors for the training

        Returns
            NotImplementedError: raises an exception if the method is not implemented by the children classes
        """
        raise NotImplementedError

    def exec_collaborative_method(self, properties, user_ratings, user_ids, movie_ids, logger):
        """
        Method should be override by the children classes. Used to execute the training, test and predictions for
        the selected model.

        Args
            properties (dict): the loaded configuration file
            user_ratings (ndarray): the input vectors for the training
            user_ids (ndarray): the "label" of each input vector
            movie_ids (ndarray): the indices related to a movie for each vector

       Returns
            NotImplementedError: raises an exception if the method is not implemented by the children classes
        """
        raise NotImplementedError

    @staticmethod
    def calc_results(properties, users, classification=Classification.binary.value):
        raise NotImplementedError
