from scipy.stats.stats import pearsonr

from models.models import User
from utils.enums import CollaborativeModels


class Pearson:

    def __init__(self):
        self.model_name = CollaborativeModels.kmeans.value
        self.metrics = {}
        self.model = None

    @staticmethod
    def init_users(user_ids, user_ratings):
        users = []
        for idx, user_id in enumerate(list(user_ids)):
            user = User(user_id, idx)
            user.user_ratings = user_ratings[idx]
            for other_idx, other_user_id in enumerate(list(user_ids)):
                # check if current and target users are different
                if user.user_id == other_user_id:
                    continue
                other_user = User(other_user_id, other_idx)
                other_user.user_ratings = user_ratings[other_idx]
                user.similar_users.append(other_user)
            users.append(user)
            return users

    def get_user_similarities(self, logger, user, user_rating, movie_ids):
        similarities = []
        absolute_similarity_list = []
        logger.info("Calculate pearson similarity with similar users")
        if not user.similar_users:
            return
        for other_user in user.similar_users:
            logger.debug("Pearson similarity with other user with id {}".format(other_user.user_id))
            other_user_ratings = other_user.user_ratings
            user_same_ratings, other_user_same_ratings, same_movie_ids = self.find_same_ratings(
                movie_ids=movie_ids, user_ratings=user_rating, other_user_ratings=other_user_ratings)
            other_user.average_user = self.get_mean_positive_ratings(other_user.user_ratings)
            if len(same_movie_ids) >= 2:
                similarity = pearsonr(user_same_ratings, other_user_same_ratings)
            else:
                similarity = (0, 0)
            logger.debug("User with id {} and user with id {} have similarity {} and in absulute value {}".
                         format(user.user_id, other_user.user_id, similarity[0], abs(similarity[0])))
            similarities.append(similarity[0])
            absolute_similarity_list.append(abs(similarity[0]))
        return similarities, absolute_similarity_list

    @staticmethod
    def get_pearson_most_similar(properties, logger, similarities, absolute_similarity_list, user):
        # sort list from min to max - pearsonr returns p-value
        num_similar_users = properties["kmeans"]["n_similar"]

        similar_users = []
        similarities_final_absolute = []
        similarities_final = []
        logger.info("Find most similar users. Number of similar users {}".format(num_similar_users))
        for i in range(num_similar_users):
            if i < len(similarities):
                max_idx = absolute_similarity_list.index(max(absolute_similarity_list))
                similarities_final_absolute.append(absolute_similarity_list[max_idx])
                similarities_final.append(similarities[max_idx])
                other_user = user.similar_users[max_idx]
                logger.debug("Most similar user: {}".format(other_user))
                logger.debug("Max similarity: {}".format(absolute_similarity_list[max_idx]))
                similar_users.append(other_user)
                absolute_similarity_list[max_idx] = 0

    @staticmethod
    def find_same_ratings(movie_ids, user_ratings, other_user_ratings):
        """
        Find the ratings of two users on the same movies.
        Args
            movie_ids (ndarray): The movie ids
            user_ratings (ndarray): The ratings of the target user
            other_user_ratings (ndarray): The ratings of a similar user to the target user

        Returns
            Three lists one with the ratings of the target and one with the ratings of the similar user and one with
            the ids of movies that both of them rated
        """
        user_same_ratings = []
        other_user_same_ratings = []
        same_movie_ids = []
        for idx, movie_id in enumerate(movie_ids):
            if user_ratings[idx] > 0 and other_user_ratings[idx] > 0:
                user_same_ratings.append(user_ratings[idx])
                other_user_same_ratings.append(other_user_ratings[idx])
                same_movie_ids.append(movie_id)
        return user_same_ratings, other_user_same_ratings, same_movie_ids

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
