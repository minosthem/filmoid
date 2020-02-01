import os

import numpy as np

import utils
from preprocessing.data_preprocessing import DataPreprocessing


class CollaborativePreprocessing(DataPreprocessing):
    users_ratings_pickle = "users_ratings.pickle"
    users_ids_pickle = "user_ids.pickle"
    users_ratings = None
    user_ids = None

    def preprocess(self, properties, datasets):
        """
        Initially, checks if the ratings list exists in the output folder and if this is the case it loads it.
        Otherwise, it takes from the ratings dataset the ratings of the users, the name of the movies from the movies
        dataset and creates a list with the movies ids. Then, within a for loop iterates the ratings dataframe and for
        each user keeps track of the ratings he gave to every movie. If he didn't rate a movie, the algorithm put a
        zero to the corresponding position of the vector. After finishing this process for every user, it returns the
        vectors of the users as a list called user_ratings and writes it to the output folder as a pickle file.

        Args
            properties (dict): dictionary with the loaded properties from the yaml file
           datasets (dict): the datasets' dictionary which was created from the read_csv function
        """
        output_folder = properties["output_folder"]
        users_ratings_pickle_filename = self.users_ratings_pickle + "_{}".format(properties["dataset"])
        users_ids_pickle_filename = self.users_ids_pickle + "_{}".format(properties["dataset"])

        if utils.check_file_exists(output_folder, users_ratings_pickle_filename):
            print("Collaborative input vectors already exist and will be loaded from pickle file")
            self.users_ratings = utils.load_from_pickle(output_folder, users_ratings_pickle_filename)
            print("Loaded user ratings of shape {}".format(self.users_ratings.shape))
        else:
            os.makedirs(output_folder, exist_ok=True)
            ratings_df = datasets["ratings"]
            movies_df = datasets["movies"]
            self.users_ratings = []
            self.user_ids = []
            movie_ids = movies_df["movieId"].values.tolist()
            print("Generating input vectors")
            for _, row in ratings_df.iterrows():
                user_id = row["userId"]
                if user_id not in self.user_ids:
                    self.user_ids.append(user_id)
                    user_ratings = ratings_df[ratings_df["userId"] == user_id]
                    user_vector = []
                    for movie_id in movie_ids:
                        rating_row = user_ratings[user_ratings["movieId"] == movie_id]
                        if not rating_row.empty:
                            rating_row = rating_row["rating"].values.tolist()
                            user_vector.append(rating_row[0])
                        else:
                            user_vector.append(0.0)
                    user_vector = np.array(user_vector)
                    self.users_ratings.append(user_vector)
                utils.print_progress(self.users_ratings)
            print("Writing input vectors into pickle file")
            self.users_ratings = np.array(self.users_ratings)
            self.user_ids = np.asarray(self.user_ids)
            utils.write_to_pickle(self.users_ratings, output_folder, users_ratings_pickle_filename)
            utils.write_to_pickle(self.user_ids, output_folder, users_ids_pickle_filename)
