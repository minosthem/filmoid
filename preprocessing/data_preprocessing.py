import pandas as pd
from nltk.tokenize import RegexpTokenizer
import utils

output_folder = "output"
input_data_pickle = "input_data.pickle"
ratings_pickle = "ratings.pickle"
users_ratings_pickle = "users_ratings.pickle"


def read_csv(files):
    datasets = {}
    for name, file in files.items():
        datasets[name] = pd.read_csv(file)
    return datasets


def preprocessing_collaborative(datasets):
    users_ratings = [[]]

    if utils.check_file_exists(output_folder, users_ratings_pickle):
        users_ratings = utils.load_from_pickle(output_folder, users_ratings_pickle)
    else:
        ratings_df = datasets["ratings"]
        movies_df = datasets["movies"]
        user_ids = []
        movie_ids = movies_df["movieId"]

        for index, row in ratings_df.iterrows():
            user_id = row["userId"]
            if user_id not in user_ids:
                user_ids.append(user_id)
                user_ratings = ratings_df[ratings_df["userId"] == user_id]
                user_vector = []
                for movie_id in movie_ids:
                    rating_row = user_ratings[user_ratings["movieId"] == movie_id]
                    user_vector.append(rating_row["rating"]) if not rating_row.empty() \
                        else user_vector.append(0)
                users_ratings.append(user_vector)
        utils.write_to_pickle(users_ratings, output_folder, users_ratings_pickle)
    return users_ratings


def preprocessing_content_based(properties, datasets):
    input_data = [[]]
    ratings = []

    if utils.check_file_exists(output_folder, input_data_pickle) and \
            utils.check_file_exists(output_folder, ratings_pickle):
        input_data = utils.load_from_pickle(output_folder, input_data_pickle)
        ratings = utils.load_from_pickle(output_folder, ratings_pickle)
    else:
        ratings_df = datasets["ratings"]
        movies_df = datasets["movies"]
        tags_df = datasets["tags"]
        glove_df = utils.load_glove_file(properties)
        for index, row in ratings_df.iterrows():
            movie_id, user_id, rating, _ = row
            # preprocess
            rating = preprocess_rating(properties, rating)
            movie_text = preprocess_text(movies_df, tags_df, movie_id, user_id)

            movie_vector = text_to_glove(properties, glove_df, movie_text)
            movie_vector.insert(0, user_id)
            # TODO standardization
            input_data.append(movie_vector)
            ratings.append(rating)
        utils.write_to_pickle(object=input_data, directory=output_folder, filename=input_data_pickle)
        utils.write_to_pickle(object=ratings, directory=output_folder, filename=ratings_pickle)
    return input_data, ratings


def text_to_glove(properties, glove_df, word_list):
    embeddings = [[]]
    for word in word_list:
        row = glove_df[glove_df.iloc[:, 0] == word]
        if not row.empty():
            vector = row[, 1:]
            vector = vector.values.tolist()
            embeddings.append(vector)
    return [sum(col) / len(col) for col in zip(*embeddings)] if properties["aggregation"] == "avg" \
        else [max(col) for col in zip(*embeddings)]


def preprocess_text(movies_df, tags_df, movie_id, user_id):
    m = movies_df[movies_df["movieId"] == movie_id]
    tags = tags_df[tags_df["userId"] == user_id and tags_df["movieId"] == movie_id]
    movie_title = m["title"]
    movie_genres = m["genres"]
    tag = tags["tag"] if not tags.empty() else ""
    # preprocessing title, genres, tags ==> remove symbols, numbers
    tokenizer = RegexpTokenizer(r'\w+')
    movie_text = movie_title + " " + movie_genres + " " + tag
    return tokenizer.tokenize(movie_text)


def preprocess_rating(properties, rating):
    if properties["classification"] == "binary":
        return 0 if rating > 3 else 1
    else:
        return int(rating)
