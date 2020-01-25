import csv
import os
import pickle
import subprocess
import sys
import time
from os.path import join, exists

import pandas as pd
import yaml

properties_folder = join(os.getcwd(), "properties")
example_properties_file = join(properties_folder, "example_properties.yaml")
properties_file = join(properties_folder, "properties.yaml")
ml_latest_small_folder = "ml-latest-small"
ml_latest = "ml-latest"


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
    # dataset_path = join(base_path, ml_latest_small_folder) if prop["dataset"] == "small" \
    #     else join(base_path, ml_latest)
    dataset_path = join(base_path, "ml-dev")
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
    :param directory: the directory where the file should be stored
    :param filename: the name of the file to be checked
    :return: boolean true if the file exists otherwise false
    """
    path = join(os.getcwd(), directory, filename)
    return exists(path)


def write_to_pickle(obj, directory, filename):
    """
    Writes an object in pickle format in the output directory
    :param obj: object to be stored in the file
    :param directory: output folder
    :param filename: the name of the file where this object is saved
    :return: does not return anything
    """
    path = join(os.getcwd(), directory, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(directory, file):
    """
    Loads an object from a pickle file.
    :param directory: output folder
    :param file: the file name to be loaded from the output folder
    :return: the object loaded from the pickle file
    """
    path = join(os.getcwd(), directory, file)
    with open(path, "rb") as f:
        return pickle.load(f)
