from os import mkdir
from os.path import join, exists

import keras
import numpy as np
from keras import callbacks
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD

from models.classifiers import ContentBasedClassifier
from utils import utils
from utils.enums import MetricKind, Classification, ContentBasedModels


class DeepNN(ContentBasedClassifier):
    """
    Class representing a Deep Neural Network.
    """

    models = []
    fold_metrics = []
    avg_metrics = {}
    test_metrics = {}
    best_model = None
    model_name = ""
    classification = ""

    def __init__(self):
        self.model_name = ContentBasedModels.dnn.value
        self.classification = Classification.binary.value

    def train(self, properties, input_data, labels):
        """
        Takes as input the vectors created from word2vec and feed them to the neural network to train it.

        Args
            properties (dict): classification and dnn parameters
            input_data (ndarray): vectors created for training
            labels (ndarray): true labels of the input data
        """
        input_dim = input_data.shape[1]
        dnn = self._build_model(properties, input_dim)
        self.models.append(dnn)
        self.classification = properties["classification"]
        num_classes = 2 if properties["classification"] == Classification.binary.value else 5
        if self.classification == Classification.multi.value:
            labels = self.change_labels(labels=labels)
        labels = keras.utils.to_categorical(labels, num_classes=num_classes)
        dnn.fit(input_data, labels, epochs=properties["dnn"]["epochs"], batch_size=properties["dnn"]["batch_size"],
                verbose=True, callbacks=self._get_training_callbacks(properties))

    def test(self, test_data, true_labels, kind=MetricKind.validation.value):
        """
        Takes the test vectors and the corresponding true labels and tests the performance of the model.

        Args
            test_data (ndarray): testing dataset
            true_labels (ndarray): testing labels
            kind (str): validation or test

        Returns
            confusion_matrix: the confusion matrix of the testing
        """
        predicted_labels = []
        result = self.models[-1].predict(test_data) if kind == MetricKind.validation.value else \
            self.best_model.predict(test_data)
        for i in range(result.shape[0]):
            predicted_labels.append(np.argmax(result[i]))
        if self.classification == Classification.multi.value:
            predicted_labels = self.change_labels(predicted_labels).tolist()
        return true_labels, predicted_labels

    @staticmethod
    def _build_model(properties, input_dim):
        """
        Builds the deep neural network by adding hidden layers and the output layer depending on the classification
        approach. Finally, it optimizes the model using stochastic gradient descent.

        Args
            properties (dict): dnn characteristics and classification
            input_dim (int): the dimension of the input vectors

        Returns
            Sequential: the neural network
        """
        dnn = Sequential()
        for i, hidden in enumerate(properties["dnn"]["hidden_layers"]):
            if type(hidden[0]) == int:
                hidden_units = hidden[0]
                activation = hidden[1]
            else:
                if properties["classification"] == Classification.binary.value:
                    hidden_units = 1 if properties["dnn"]["loss"] == "binary_crossentropy" else 2
                else:
                    hidden_units = 5
                activation = hidden[0]
            if i == 0:
                dnn.add(Dense(hidden_units, activation=activation, input_dim=input_dim))
            else:
                dnn.add(Dense(hidden_units, activation=activation))
            if len(hidden) == 3:
                dnn.add(Dropout(hidden[2]))
        sgd = SGD(lr=properties["dnn"]["sgd"]["lr"], decay=properties["dnn"]["sgd"]["decay"],
                  momentum=properties["dnn"]["sgd"]["momentum"], nesterov=properties["dnn"]["sgd"]["nesterov"])

        dnn.compile(loss=properties["dnn"]["loss"], optimizer=sgd, metrics=properties["dnn"]["metrics"])
        return dnn

    @staticmethod
    def _get_training_callbacks(properties, monitor_metric="val_loss"):
        """
        It saves the model after some period, stops it if there is no improvement after a predetermined number of epochs
        and modifies the learning rate if the improvement of the loss function is smaller than a threshold value.

        Args
            properties (dict): output folder and dnn characteristics
            monitor_metric (str): validation loss which is the monitoring measure for the callbacks

        Returns
            list: callbacks after the termination of the model training
        """
        output_folder = join(utils.app_dir, properties["output_folder"])
        model_folder_path = join(output_folder, "model")
        if not exists(model_folder_path):
            mkdir(model_folder_path)
        early_stopping_patience = int(properties["dnn"]["epochs"] / 2)
        reduce_lr = int(properties["dnn"]["epochs"] / 5)
        callbacks_list = [
            callbacks.ModelCheckpoint(join(model_folder_path, "model_weights.{epoch:02d}-{loss:.2f}.hdf5"),
                                      monitor=monitor_metric, verbose=0, save_weights_only=False,
                                      mode='auto', period=1),
            callbacks.EarlyStopping(monitor=monitor_metric, min_delta=0, patience=early_stopping_patience,
                                    verbose=0, mode='auto', baseline=None, restore_best_weights=False),
            callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.1,
                                        patience=reduce_lr, verbose=0, mode='auto',
                                        min_delta=0.0001, cooldown=0, min_lr=0),
            keras.callbacks.callbacks.CSVLogger(
                join(utils.app_dir, properties["output_folder"], "logs", utils.log_filename + ".log"), separator=',',
                append=True)
        ]

        return callbacks_list

    @staticmethod
    def change_labels(labels):
        new_labels = []
        for label in labels:
            label = label - 1
            new_labels.append(label)
        return np.asarray(new_labels)
