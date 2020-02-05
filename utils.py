import csv
import logging
import os
import pickle
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from os import mkdir, chdir, remove
from os.path import join, exists

import matplotlib.pyplot as plt
import pandas as pd
import wget
import yaml

from enums import MetricCaptions, Datasets
from preprocessing.data_preprocessing import DataPreprocessing

log_filename = 'logs_%s' % datetime.now().strftime('%Y%m%d-%H%M%S')
current_dir = os.getcwd()


def config_logger():
    """
    Creates a logger which writes both in a file and in the console

    Returns
        Logger: the logger object
    """
    if not exists(join(current_dir, "logs")):
        os.mkdir(join(current_dir, "logs"))

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    program_logger = logging.getLogger(__name__)

    program_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("{0}/{1}.log".format("logs", log_filename))
    file_handler.setFormatter(log_formatter)
    program_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    program_logger.addHandler(console_handler)
    return program_logger


logger = config_logger()


def print_progress(container, step=20, msg="\tProcessed {} elements."):
    """
    Function to periodically print the progress of the flow

    Args:
        container (list): list or nd array containing the input data
        step (int): integer value based on which the printing period is defined
        msg (str): message to be printed

    Returns:
        bool: The return value. True for success, False otherwise.
    """
    if len(container) % step == 0 and container:
        logger.info(msg.format(len(container)))


def setup_folders(properties):
    """
    Function to set up the necessary folders and files. Based on the operating system, it uses the relevant script
    (setup.sh or setup.bat) to download the datasets and the word embeddings from glove.

    Args:
        properties (dict): dictionary containing all the loaded properties from the respective yaml file
    """
    if properties["os"] == "python":
        create_missing_directories(properties["output_folder"], properties["resources_folder"],
                                   properties["datasets_folder"])
        download_embeddings_file(properties["resources_folder"], properties["embeddings_file_url"])
        download_datasets(properties["datasets_folder"])
    else:
        os.environ["OUTPUT_FOLDER"] = properties["output_folder"]
        os.environ["DATASETS_FOLDER"] = properties["datasets_folder"]
        os.environ["RESOURCES_FOLDER"] = properties["resources_folder"]
        os.environ["EMBEDDINGS_ZIP_FILE"] = properties["embeddings_zip_file"]
        os.environ["EMBEDDINGS_FILE_URL"] = properties["embeddings_file_url"]
        process = None
        if properties["os"] == "linux":
            file_path = join(os.getcwd(), "setup", "setup.sh")
            process = subprocess.Popen("bash {}".format(file_path), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       shell=True)
        elif properties["os"] == "windows":
            file_path = join(os.getcwd(), "setup", "setup.bat")
            process = subprocess.Popen(file_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        process.wait()
        output, error = process.stdout, process.stderr
        if process:
            if output:
                logger.info("Folders setup finished successfully!")
            if error:
                logger.info("Folders setup failed because: {}".format(error))
                exit(-1)


def create_missing_directories(output_folder, resources_folder, datasets_folder):
    print("Create missing directories")
    if not exists(output_folder):
        mkdir(output_folder)
    if not exists(resources_folder):
        mkdir(resources_folder)
    if not exists(datasets_folder):
        mkdir(datasets_folder)


def download_embeddings_file(resources_folder, embeddings_file_url):
    # download embeddings file
    chdir(resources_folder)
    print("Download specified embeddings file")
    embeddings_zip_file = wget.download(embeddings_file_url)
    path_to_embeddings_file = join(resources_folder, embeddings_zip_file)
    with zipfile.ZipFile(path_to_embeddings_file, 'r') as zip_ref:
        zip_ref.extractall(resources_folder)
    remove(path_to_embeddings_file)


# download datasets
def download_datasets(datasets_folder):
    chdir(datasets_folder)
    print("Download MovieLens datasets")
    small_dataset = wget.download("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")
    large_dataset = wget.download("http://files.grouplens.org/datasets/movielens/ml-latest.zip")
    path_to_small_dataset = join(datasets_folder, small_dataset)
    path_to_large_dataset = join(datasets_folder, large_dataset)

    with zipfile.ZipFile(path_to_small_dataset, 'r') as zip_ref:
        zip_ref.extractall(datasets_folder)

    with zipfile.ZipFile(path_to_large_dataset, 'r') as zip_ref:
        zip_ref.extractall(datasets_folder)

    remove(path_to_small_dataset)
    remove(path_to_large_dataset)


def elapsed_str(previous_tic, up_to=None):
    """
    Calculates the time passed from a previous tic.

    Args:
        previous_tic (float): time in float format of some previous time
        up_to (float): define the end time - if not provided then current time will be calculated

    Returns:
        str: The return value. Time difference in str format
    """
    if up_to is None:
        up_to = time.time()
    duration_sec = up_to - previous_tic
    m, s = divmod(duration_sec, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def load_properties():
    """
    Load yaml file containing program's properties. Creates the output, resources and datasets folders if they  do not
    exist. Check if the resources and datasets folders contain files/folders, otherwise checks if the setup folder
    property is set to false and the output directory is empty.
    If all conditions are true the program prints a warning and stops, because datasets and embeddings are missing (both
    as files or pickle stored objects).

    Check the ReadMe.md file of the directory for a complete list of the possible configuration properties and their
    values.

    Returns:
        dict: the properties dictionary
    """
    properties_folder = join(os.getcwd(), "properties")
    example_properties_file = join(properties_folder, "example_properties.yaml")
    properties_file = join(properties_folder, "properties.yaml")
    file = properties_file if exists(properties_file) else example_properties_file
    with open(file, 'r') as f:
        properties = yaml.safe_load(f)
        output_folder = join(os.getcwd(), properties["output_folder"]) if properties["output_folder"] else \
            join(os.getcwd(), "output")
        resources_folder = join(os.getcwd(), properties["resources_folder"]) if properties["resources_folder"] else \
            join(os.getcwd(), "resources")
        datasets_folder = join(os.getcwd(), properties["datasets_folder"]) if properties["datasets_folder"] else \
            join(os.getcwd(), "Datasets")
        if not exists(output_folder):
            os.mkdir(output_folder)
        if not exists(resources_folder):
            os.mkdir(resources_folder)
        if not exists(datasets_folder):
            os.mkdir(datasets_folder)
        if not os.listdir(resources_folder) and not properties["setup_folders"] and not os.listdir(output_folder):
            logger.info(
                "Resources folder is empty and setup folders property is set to false. You need to either download "
                "manually an embeddings file or set the property to true")
            exit(-1)
        if not os.listdir(datasets_folder) and not properties["setup_folders"] and not os.listdir(output_folder):
            logger.info(
                "Datasets folder is empty and setup folders property is set to false. You need to either download "
                "manually the MovieLens educational datasets or set the property to true")
            exit(-1)
    return properties


def get_filenames(prop):
    """
    Creates the base path for the datasets files as well as the path for each of them.

    Args:
        prop (dict): filenames, dataset-file-extention

    Returns:
        dict: the files dictionary and each dataset's path
    """
    files = {}
    datasets_folder = prop["datasets_folder"]
    base_path = join(os.getcwd(), datasets_folder)
    dataset_path = join(base_path, Datasets.ml_latest_small.value) if prop["dataset"] == Datasets.small.value \
        else join(base_path, Datasets.ml_latest.value)
    for file in prop["filenames"]:
        filename = file + prop["dataset-file-extention"]
        files[file] = join(dataset_path, filename)
    return files


def load_glove_file(properties):
    """
    Creates the glove's file path and reads this file.

    Args:
        properties (dict): embeddings_file

    Returns:
        DataFrame: glove word embeddings as Dataframe
    """
    max_int = sys.maxsize
    resources_folder = join(os.getcwd(), properties["resources_folder"])
    glove_file_path = join(os.getcwd(), properties["resources_folder"], properties["embeddings_file"])
    if not exists(resources_folder):
        logger.info("Resources folder not found in", resources_folder)
        return None
    if not exists(resources_folder):
        logger.info("Glove file not found in", glove_file_path)
        return None
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    res = pd.read_csv(glove_file_path, index_col=0, delimiter=" ", quoting=3, header=None, engine="python",
                      error_bad_lines=False)
    return res


def check_file_exists(directory, filename):
    """
    Checks if the the path of a file exists in order not to create it again.

    Args:
        directory (str): the directory where the file should be stored
        filename (str): the name of the file to be checked

    Returns:
        bool: True if the file exists otherwise False
    """
    path = join(os.getcwd(), directory, filename)
    return exists(path)


def write_to_pickle(obj, directory, filename):
    """
    Writes an object in pickle format in the output directory

    Args:
        obj (Object): the object to be stored in a pickle file
        directory (str): the directory where the file should be stored
        filename (str): the name of the file where the object will be written
    """
    path = join(os.getcwd(), directory, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(directory, file):
    """
    Loads an object from a pickle file.

    Args:
        directory (str): the directory where the file should be stored
        file(str): the name of the file where the object will be written

    Returns:
        Object: the object loaded from the pickle file
    """
    path = join(os.getcwd(), directory, file)
    with open(path, "rb") as f:
        return pickle.load(f)


def visualize(df, output_folder, results_folder, folder_name, filename):
    """
    Method to visualize the results of a classifier for a specific fold, avg folds or test
    Saves the plot into the specified folder and filename.

    Args
        df (DataFrame): a pandas DataFrame with the results of the model
        output_folder (str): the name of the output folder
        results_folder (str): the name of the folder of the current execution
        folder_name (str): the fold name or avg or test folder
        filename (str): the name of the figure
    """
    captions = [MetricCaptions.macro_prec.value, MetricCaptions.micro_prec.value, MetricCaptions.macro_recall.value,
                MetricCaptions.micro_recall.value, MetricCaptions.macro_f.value, MetricCaptions.micro_f.value]
    df.pivot("classifier", "metric", "result").plot(kind='bar', width=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(rotation="horizontal")
    plt.legend(captions)
    plt.savefig(join(output_folder, results_folder, folder_name, filename))
    # plt.show()


def generate_recommendation_dataset():
    """
    It creates a dataset with the movies that the users have not rated and saves it in a csv file in the dataset folder.

    """
    properties = load_properties()
    file_names = get_filenames(properties)
    # read datasets
    logger.info("Creating dataset for recommendation")
    dp = DataPreprocessing()
    dp.read_csv(file_names)
    csvs = dp.datasets
    ratings = csvs["ratings"]
    movies = csvs["movies"]
    user_ids = []
    test_df = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    count = 0
    for _, row in ratings.iterrows():
        user_id = row["userId"]
        if user_id in user_ids:
            continue
        user_ids.append(user_id)
        logger.info("Checking for user with id {}".format(row["userId"]))
        user_ratings = ratings[ratings["userId"] == user_id]
        movie_ids = movies["movieId"]
        for movie_id in movie_ids:
            if movie_id not in user_ratings["movieId"].values.tolist():
                logger.info("Adding line number {}".format(count))
                logger.info("Adding row with user id {} and movie id {}".format(user_id, movie_id))
                test_df.loc[count] = [user_id, movie_id, None, datetime.now()]
                count += 1
    dataset_folder = Datasets.ml_latest.value if properties["dataset"] == Datasets.latest.value else \
        Datasets.ml_latest_small.value
    path_to_dataset = join(os.getcwd(), properties["datasets_folder"], dataset_folder)
    file_path = join(path_to_dataset, "test_recommendation.csv")
    test_df.to_csv(file_path, sep=",")
