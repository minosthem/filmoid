from os.path import join

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

from models.kmeans import Kmeans
from models.pearson import Pearson
from utils import utils
from utils.enums import Classification
from utils.enums import MetricNames, MetricKind, CollaborativeModels


class CollaborativeMethod:
    """
    Class to be extended by collaborative models (e.g. kmeans). Methods train and test should be implemented by the
    sub-classes.
    """

    metrics = {}
    model_name = CollaborativeModels.collaborative.value

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
        pearson = Pearson()
        if CollaborativeModels.kmeans.value in properties["models"]["collaborative"]:
            kmeans = Kmeans()
            predictions = kmeans.fit_transform(properties=properties, input_data=user_ratings)
            users = kmeans.find_similar_users(user_ids=user_ids, user_ratings=user_ratings, predictions=predictions)
        else:
            users = pearson.init_users(user_ids=user_ids, user_ratings=user_ratings)
        for user in users:
            logger.info("Calculating predictions for user with id {}".format(user.user_id))
            user_rating = user_ratings[user.user_idx, :]
            user.average_rating = self.get_mean_positive_ratings(user_rating)
            logger.info("User with id {} has average rating {}".format(user.user_id, user.average_rating))
            similarities, absolute_similarity_list = pearson.get_user_similarities(logger=logger, user=user,
                                                                                   user_rating=user_rating,
                                                                                   movie_ids=movie_ids)
            if similarities and absolute_similarity_list:
                similar_users, similarities_final_absolute, similarities_final = pearson.get_pearson_most_similar(
                    properties=properties, logger=logger, similarities=similarities,
                    absolute_similarity_list=absolute_similarity_list, user=user)
                self.get_user_predictions(logger=logger, movie_ids=movie_ids, similar_users=similar_users,
                                          similarities_final=similarities_final, user=user,
                                          similarities_final_absolute=similarities_final_absolute,
                                          properties=properties)
        logger.info("Save all results to pickle file")
        utils.write_to_pickle(users, properties["output_folder"],
                              "collaborative_user_predictions_{}.pickle".format(properties["dataset"]))
        return users

    def get_user_predictions(self, logger, movie_ids, similar_users, similarities_final, similarities_final_absolute,
                             user, properties):
        """
        Calculates the prediction ratings of every user for evey movie based on similar users. Then, keeps track of the
        true and predicted ratings of the users and writes them into a csv file.

        Args
            logger (Logger): handles the logs and prints
            movie_ids (list): the list of movies' ids
            similar_users (list): the list of similar users to a target user
            similarities_final (list): the list with the similarity coefficients of similar users
            similarities_final_absolute (list): the list with the absolute similarity coefficients of similar users
            user (User): a user object
            properties (dict): output folder, dataset
        """
        logger.info("Calculate user predictions for all movies")
        user.movie_predictions = []
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
        self.write_user_csv(properties, user, movie_ids)

    def calc_results(self, properties, users, logger, classification=Classification.binary.value):
        """
            Calculates the average value of the evaluation metrics of the users' predicted ratings and writes the
            results in a csv file.

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

    @staticmethod
    def get_mean_positive_ratings(ratings):
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

    def write_user_csv(self, properties, user, movie_ids):
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
            # rating = 0 if rating > 3 else 1
            prediction = user.movie_predictions[movie_idx]
            # prediction = 0 if prediction > 3 else 1
            df.loc[movie_idx] = [movie_id, rating, prediction]
        file_path = join(path, filename)
        df.to_csv(file_path, sep=',')
