import csv
import os
import pickle
import subprocess
import sys
import time
from os.path import join, exists
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from models.knn_classifier import KNN
from models.rf_classifier import RandomForest
from models.dnn_classifier import DeepNN

properties_folder = join(os.getcwd(), "properties")
example_properties_file = join(properties_folder, "example_properties.yaml")
properties_file = join(properties_folder, "properties.yaml")
ml_latest_small_folder = "ml-latest-small"
ml_latest = "ml-latest"
metric_names = ["macro-precision", "micro-precision", "macro-recall", "micro-recall", "macro-f", "micro-f"]


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
        print(msg.format(len(container)))


def setup_folders(properties):
    """
    Function to set up the necessary folders and files. Based on the operating system, it uses the relevant script
    (setup.sh or setup.bat) to download the datasets and the word embeddings from glove.

    Args:
        properties (dict): dictionary containing all the loaded properties from the respective yaml file
    """
    os.environ["OUTPUT_FOLDER"] = properties["output_folder"]
    os.environ["DATASETS_FOLDER"] = properties["datasets_folder"]
    os.environ["RESOURCES_FOLDER"] = properties["resources_folder"]
    embeddings_file = properties["embeddings_zip_file"]
    os.environ["EMBEDDINGS_ZIP_FILE"] = embeddings_file
    os.environ["EMBEDDINGS_FILE_URL"] = properties["embeddings_file_url"]
    process = None
    if properties["os"] == "linux":
        process = subprocess.Popen("bash setup.sh", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    elif properties["os"] == "windows":
        process = subprocess.Popen("./setup.bat", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    process.wait()
    output, error = process.stdout, process.stderr
    if process:
        if output:
            print("Folders setup finished successfully!")
        if error:
            print("Folders setup failed because: {}".format(error))
            exit(-1)


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
    Load yaml file containing program's properties.
    
    - os: define your operating system. The possible values that are currently handled by the software are linux and windows
    - setup_folders: define whether you want to download the input data (glove file and datasets). If the files are already downloaded inside the Datasets and resources directories, set this parameter as False.
    - output_folder: the name of the output folder where the saved models will be stored as well as the results
    - datasets_folder: the name of the datasets directory
    - resources_folder: the name of the directory where the word embeddings file will be stored
    - embeddings_file: the name of the word embeddings file to be used(e.g. "glove.6B.50d.txt")
    - embeddings_zip_file: the zip file name to extract when downloaded (e.g. "glove.6B.zip")
    - embeddings_file_url: the URL from where to download the word embeddings file (e.g. "http://nlp.stanford.edu/data/glove.6B.zip")
    - dataset: the dataset to be used in the experiments. Currently the possible values are small and latest
    - filenames: the csv names which you should provide as follows: ["links", "movies", "ratings", "tags"]
    - dataset-file-extention: the file extension of the dataset files. You should provide it: ".csv"
    - methods: the recommendation policies which are: ["collaborative", "content-based"]
    - models: the classifiers to be used. Choose some or all of the following: ["kmeans", "knn", "rf", "dnn"]
    - aggregation: word embeddings aggregation strategies. Possible strategies: avg, max
    - classification: binary or multi-class classification. Possible values: binary, multi

    Returns:
        dict: the properties dictionary
    """
    file = properties_file if exists(properties_file) else example_properties_file
    with open(file, 'r') as f:
        properties = yaml.safe_load(f)
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
    dataset_path = join(base_path, ml_latest_small_folder) if prop["dataset"] == "small" \
        else join(base_path, ml_latest)
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
        print("Resources folder not found in", resources_folder)
        return None
    if not exists(resources_folder):
        print("Glove file not found in", glove_file_path)
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


def init_classifier(classifier_name):
    """
    Function that inits a classifier object based on a given name. Stores the given name in a field of the object.

    Args
        classifier_name (str): the name of the model

    Returns
        Classifier: a classifier object
    """
    classifier = None
    if classifier_name == "knn":
        classifier = KNN()
    elif classifier_name == "rf":
        classifier = RandomForest()
    elif classifier_name == "dnn":
        classifier = DeepNN()
    if classifier:
        classifier.model_name = classifier_name
    return classifier


def visualize(df, output_folder, results_folder, folder_name, filename):
    df.pivot("classifier", "metric", "result").plot(kind='bar')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(join(output_folder, results_folder, folder_name, filename))
    plt.show()
