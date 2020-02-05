import pandas as pd

from enums import PreprocessKind


class DataPreprocessing:
    """
    Generic class that represents a data preprocessing procedure. A class that extends the DataPreprocessing class
    should override and implement the preprocess method as well as the dataset creation and k-fold cross-validation
    methods.
    """
    datasets = {}

    def read_csv(self, filenames):
        """
        Reads every dataset from the files dictionary.Stores the datasets into a class field (dictionary).

        Args
            filenames (dict): the path to the dataset csv files
        """
        for name, file in filenames.items():
            self.datasets[name] = pd.read_csv(file)

    def preprocess(self, properties, datasets, kind=PreprocessKind.train.value):
        """
        Method that should be override by the children classes. Implements the preprocessing step of the method.

        Args
            properties (dict): the loaded configuration file
            datasets (dict): the DataFrames with the links, movies, tags and ratings
            kind (str): values can be either train or recommend - input dataset for ratings will be different

        Returns
            NotImplementedError: raises an exception if the child class has not implemented the method
        """
        raise NotImplementedError

    def create_train_test_data(self, input_data, labels):
        """
        Method implemented by the content based preprocessing child class. Splits the dataset into training and test
        sets.

        Args
            input_data (ndarray): the input vectors of the dataset
            labels (ndarray): the labels of the instances of the dataset
        """
        pass

    def create_cross_validation_data(self, input_data, properties):
        """
        Method implemented by the content based preprocessing child class. Creates the k-fold cross-validation indices.

        Args
            input_data (ndarray): the input vectors of the dataset
            properties (dict): the loaded configuration file
        """
        pass
