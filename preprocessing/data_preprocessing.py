import re
import string

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import utils

input_data_pickle = "input_data.pickle"
ratings_pickle = "ratings.pickle"
users_ratings_pickle = "users_ratings.pickle"

punct_digit_to_space = str.maketrans(string.punctuation + string.digits, " " * len(string.punctuation + string.digits))


def read_csv(files):
    """
    Reads every dataset from the files dictionary.
    :param files: files dictionary
    :return: a dictionary containing all the loaded Dataframes
    """
    datasets = {}
    for name, file in files.items():
        datasets[name] = pd.read_csv(file)
    return datasets


def create_train_test_data(input, labels):
    input_train, input_test, labels_train, labels_test = train_test_split(input, labels,
                                                                          test_size=0.2,
                                                                          random_state=0)
    return input_train, input_test, labels_train, labels_test


def create_cross_validation_data(input, properties):
    kf = KFold(n_splits=properties["cross-validation"], shuffle=True, random_state=666)
    return kf.split(input)


def preprocessing_collaborative(properties, datasets):
    """
    Initially, checks if the ratings list exists in the output folder and if this is the case it loads it. Otherwise,
    it takes from the ratings dataset the ratings of the users, the name of the movies from the movies dataset and
    creates a list with the movies ids. Then, within a for loop iterates the ratings dataframe and for each user
    keeps track of the ratings he gave to every movie. If he didn't rate a movie, the algorithm put a zero to the
    corresponding position of the vector. After finishing this process for every user, it returns the vectors of the
    users as a list called user_ratings and writes it to the output folder as a pickle file.

    :param properties: dictionary with the loaded properties from the yaml file
    :param datasets: the datasets' dictionary which was created from the read_csv function
    :return: the user_ratings list which is a list of vectors containing the ratings of every user
    """
    users_ratings = [[]]
    output_folder = properties["output_folder"]
    if utils.check_file_exists(output_folder, users_ratings_pickle):
        print("Collaborative input vectors already exist and will be loaded from pickle file")
        users_ratings = utils.load_from_pickle(output_folder, users_ratings_pickle)
    else:
        ratings_df = datasets["ratings"]
        movies_df = datasets["movies"]
        user_ids = []
        movie_ids = movies_df["movieId"].values.tolist()
        print("Generating input vectors")
        for index, row in ratings_df.iterrows():
            user_id = row["userId"]
            if user_id not in user_ids:
                user_ids.append(user_id)
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
                users_ratings.append(user_vector)
        print("Writing input vectors into pickle file")
        users_ratings = np.array(users_ratings)
        users_ratings_pickle_filename = users_ratings_pickle + "_{}".format(properties["dataset"])
        utils.write_to_pickle(users_ratings, output_folder, users_ratings_pickle_filename)
    return users_ratings


def preprocessing_content_based(properties, datasets):
    """
    Checks if the input and the rating file exist and loads them from the output folder. Otherwise, takes the rating,
    movies and tags datasets and converts them to dataframes and also loads the glove file. It iterates the ratings
    dataframe keeping from every row the movie id, user id and the rating. It uses the functions preprocess_rating,
    preprocess_text and text_to_glove to create a vector corresponding to a movie's features and user id. The user's id
    is added on the first position of that vector. Every vector is added to a list of vectors called input_data.
    Finally, the rating of every user for a particular movie is added to a list called ratings and both this list as
    well as the input_data list are being saved to the output folder.

    :param properties: embedding file, classification, aggregation
    :param datasets: dictionary containing the dataframes of all the movielens csvs
    :return: the vectors of numbers for the movies containing the tags of a user and the ratings list
    """
    input_data = []
    ratings = []
    output_folder = properties["output_folder"]
    if utils.check_file_exists(output_folder, input_data_pickle) and \
            utils.check_file_exists(output_folder, ratings_pickle):
        print("Content-based input data already exist and will be loaded from pickle file")
        input_data = utils.load_from_pickle(output_folder, input_data_pickle)
        ratings = utils.load_from_pickle(output_folder, ratings_pickle)
    else:
        ratings_df = datasets["ratings"]
        movies_df = datasets["movies"]
        tags_df = datasets["tags"]
        glove_df = utils.load_glove_file(properties)
        print("Generating input vectors")
        for index, row in ratings_df.iterrows():
            movie_id, user_id, rating, _ = row
            # preprocess
            rating = preprocess_rating(properties, rating)
            movie_text = preprocess_text(movies_df, tags_df, movie_id, user_id)

            movie_vector = text_to_glove(properties, glove_df, movie_text)
            if movie_vector.size == 0:
                continue
            movie_vector = np.insert(movie_vector, 0, user_id,axis=1)
            input_data.append(movie_vector)
            ratings.append(rating)
            # limit data size for testing purposes
            if "limit_input" in properties and properties["limit_input"] <= len(ratings):
                break
            if len(ratings) % 20 == 0 and ratings:
                print("Processed {} inputs.".format(len(ratings)))

        input_data = np.concatenate(input_data)
        print("Produced a feature matrix of shape {}".format(input_data.shape))
        # standardization
        print("Standardize input vectors")
        input_data = preprocessing.scale(input_data)
        print("Save input vectors to file")
        input_data_pickle_filename = input_data_pickle + "_{}".format(properties["dataset"])
        ratings_pickle_filename = ratings_pickle + "_{}".format(properties["dataset"])
        utils.write_to_pickle(obj=input_data, directory=output_folder, filename=input_data_pickle_filename)
        utils.write_to_pickle(obj=ratings, directory=output_folder, filename=ratings_pickle_filename)
    return input_data, ratings


def text_to_glove(properties, glove_df, word_list):
    """
    Takes the pre-processed text created by the preprocess_text function and converts it to a vector of numbers using
    the word embeddings from Glove (https://nlp.stanford.edu/projects/glove/).
    This process is being done for every word of the text separately. In the end all the vectors
    are gathered in a list of vectors (embeddings) and, depending on the aggregation policy defined in the properties
    yaml file, it creates a single vector containing the average or the maximum number of every position of all the
    word embeddings vectors.

    :param properties: aggregation strategy (avg or max)
    :param glove_df: embedding file
    :param word_list: movie text or word list
    :return: a vector for every movie text which contains the movie title and genre and also the tags of a user for
    a specific movie
    """
    embeddings = []
    for word in word_list:
        # to lowercase
        word = word.lower()
        if word not in glove_df.index:
            continue
        # expand dimensions to a 1 x <dim> vector
        embeddings.append(np.expand_dims(glove_df.loc[word].values, 0))

    # concatenate to a <num words> x <dimension> matrix
    embeddings = np.concatenate(embeddings, axis=0)

    # aggregate all word vectors to a single vector
    if properties["aggregation"] == "avg":
        embeddings = np.mean(embeddings, axis=0)
    elif properties["aggregation"] == "max":
        embeddings = np.max(embeddings, axis=0)

    # expand dimension to a 1 x <dim> vector
    return np.expand_dims(embeddings, 0)


def preprocess_text(movies_df, tags_df, movie_id, user_id):
    """
    It keeps the movie id from the movies dataset and also the tags of a particular user for the specific movie. Then,
    creates a string containing the movie title and movie genre which are taken from the movies dataset as well as the.
    Finally, it adds the tags of the user for this movie to the string and gets rid of every symbol and number of that
    text.

    :param movies_df: movies dataset
    :param tags_df: tags dataset
    :param movie_id: id of a movie
    :param user_id: id of a user
    :return: the processed text containing the title and genre of a movie and the tags of a user for this movie
    """
    m = movies_df[movies_df["movieId"] == movie_id]
    tags = tags_df[(tags_df["userId"] == user_id) & (tags_df["movieId"] == movie_id)]
    movie_title = m.iloc[0]["title"]
    movie_genres = m.iloc[0]["genres"]
    movie_text = movie_title + " " + movie_genres
    if not tags.empty:
        tags = tags["tag"]
        for index, row in tags.iterrows():
            movie_text = movie_text + " " + row["tag"]
    # preprocessing title, genres, tags ==> remove symbols, numbers
    # remove digits and punctuation
    movie_text = movie_text.translate(punct_digit_to_space).strip()
    # merge multiple spaces
    movie_text = re.sub("[ ]+", " ", movie_text)
    # tokenization with the class below is probably not necessary
    # tokenize
    # tokenizer = RegexpTokenizer(r'\w+')
    # return tokenizer.tokenize(movie_text)
    return movie_text.split()


def preprocess_rating(properties, rating):
    """
    Converts a rating to a binary score o and 1 if the classification policy is binary else it keeps the rating and
    rounds it (uses the ratings as Integer numbers).

    :param properties: classification
    :param rating: rating of user for a movie
    :return: a binary rating
    """
    if properties["classification"] == "binary":
        return 0 if rating > 3 else 1
    else:
        return int(rating)
