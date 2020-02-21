class User:
    """
    Class that represents users.
    """
    user_id = -1
    user_idx = -1
    similar_users = []
    user_ratings = []
    true_rated = []
    similarities = []
    user_cluster_idx = -1
    average_rating = 0.0
    movie_predictions = []
    predicted_rated = []

    def __init__(self, user_id, user_idx):
        self.user_id = user_id
        self.user_idx = user_idx
        self.similar_users = []
        self.user_ratings = []
        self.user_cluster_idx = -1
        self.average_rating = 0.0
        self.movie_predictions = []
        self.predicted_rated = []