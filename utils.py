import os
import pickle
import subprocess
from os.path import join, exists

import pandas as pd
import yaml

properties_folder = join(os.getcwd(), "properties")
example_properties_file = join(properties_folder, "example_properties.yaml")
properties_file = join(properties_folder, "properties.yaml")
ml_latest_small_folder = "ml-latest-small"
ml_latest = "ml-latest"


def setup_folders(properties):
    """
    Sets up environmental variables in order to execute the relevant script based on the operating system.
    The script creates the output, datasets and resources folders and downloads the dataset csv files as well as
    the defined glove txt file
    :param properties: dictionary with the properties loaded from the yaml file
    :return: nothing
    """
    os.environ["OUTPUT_FOLDER"] = properties["output_folder"]
    os.environ["DATASETS_FOLDER"] = properties["datasets_folder"]
    os.environ["RESOURCES_FOLDER"] = properties["resources_folder"]
    embeddings_file = properties["embeddings_file"].replace(".txt", ".zip")
    os.environ["EMBEDDINGS_FILE"] = embeddings_file
    os.environ["EMBEDDINGS_FILE_URL"] = properties["embeddings_file_url"]
    process = None
    if properties["os"] == "linux":
        process = subprocess.Popen("bash setup.sh", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    elif properties["os"] == "windows":
        process = subprocess.Popen("./setup.bat", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    output, error = process.stdout, process.stderr
    if process:
        if output:
            print("Folders setup finished successfully!")
        if error:
            print("Folders setup failed because: {}".format(error))
            exit(-1)


def load_properties():
    """
    Load yaml file containing program's properties.
    :return: the properties dictionary
    """
    file = properties_file if exists(properties_file) else example_properties_file
    with open(file, 'r') as f:
        properties = yaml.safe_load(f)
    return properties


def get_filenames(prop):
    """
    Creates the base path for the datasets files as well as the path for each of them.
    :param prop: filenames, dataset-file-extention
    :return: the files dictionary and each dataset's path
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
    :param properties: embeddings_file
    :return: glove word embeddings as Dataframe
    """
    glove_file_path = join(os.getcwd(), properties["resources_folder"], properties["embeddings_file"])
    return pd.read_csv(glove_file_path, delimiter=" ", header=None)


def check_file_exists(directory, filename):
    """
    Checks if the the path of a file exists in order not to create it again.
    :param directory: the directory where the file should be stored
    :param filename: the name of the file to be checked
    :return: boolean true if the file exists otherwise false
    """
    path = join(os.getcwd(), directory, filename)
    return True if exists(path) else False


def write_to_pickle(obj, directory, filename):
    """
    Writes an object in pickle format in the output directory
    :param obj: object to be stored in the file
    :param directory: output folder
    :param filename: the name of the file where this object is saved
    :return: does not return anything
    """
    path = join(os.getcwd(), directory, filename)
    pickle.dump(obj, path)


def load_from_pickle(directory, file):
    """
    Loads an object from a pickle file.
    :param directory: output folder
    :param file: the file name to be loaded from the output folder
    :return: the object loaded from the pickle file
    """
    path = join(os.getcwd(), directory, file)
    return pickle.load(path)
