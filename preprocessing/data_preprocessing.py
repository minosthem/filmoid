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
        pass

    def create_train_test_data(self, input_data, labels):
        pass

    def create_cross_validation_data(self, input_data, properties):
        pass
