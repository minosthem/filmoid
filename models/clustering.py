from scipy.stats.stats import pearsonr

import utils


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
        for model_name in properties["models"]["collaborative"]:
            self.model_name = model_name
            model = utils.init_collaborative_model(model_name=self.model_name)
            predictions = model.fit_transform(properties=properties, input_data=user_ratings)
            users = self.__find_similar_users(user_ids=user_ids, user_ratings=user_ratings, predictions=predictions)
            for user in users:
                user_rating = user_ratings[user.user_idx, :]
                similarities = []
                for other_user in user.similar_users:
                    other_user_ratings = user_ratings[other_user[1], :]
                    similarity = pearsonr(user_rating, other_user_ratings)
                    similarities.append(similarity)
                # sort list from min to max - pearsonr returns p-value
                num_similar_users = properties["kmeans"]["n_similar"]
                similar_users = []
                for i in range(num_similar_users):
                    min_idx = similarities.index(min(similarities))
                    other_user = user.similar_users[min_idx]
                    similar_users.append(other_user)
                    similarities[min_idx] = 100000000
                # TODO predict ratings

    @staticmethod
    def __find_similar_users(user_ids, user_ratings, predictions):
        users = []
        rows, cols = predictions.shape
        for row in range(0, rows):
            predictions[row, :].sort()
        for idx, user_id in user_ids:
            user = User(user_id, idx)
            user.user_ratings = user_ratings[idx]
            user.user_similarities = predictions[idx, 0]
            for row in range(0, rows):
                if predictions[row, 0] == user.user_similarities:
                    user.similar_users.append((user_ids[row], row))
            users.append(user)
        return users


class User:
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
