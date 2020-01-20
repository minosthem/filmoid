import os
from os.path import join, exists
import pandas as pd
import pickle
import yaml

properties_folder = join(os.getcwd(), "properties")
example_properties_file = join(properties_folder, "example_properties.yaml")
properties_file = join(properties_folder, "properties.yaml")
datasets_folder = "Datasets"
ml_latest_small_folder = "ml-latest-small"
ml_latest = "ml-latest"


def load_properties():
    """
    Load yaml file containing program's properties.

    :return: the properties dictionary and the properties folder path
    """
    file = properties_file if exists(properties_file) else example_properties_file
    with open(file, 'r') as f:
        properties = yaml.safe_load(f)
    return properties


def get_filenames(prop):
    """
    Creates the base path for the datasets files as well as the path for each of them.

    :param prop: filenames, dataset-file-extention
    :return: the files dictionary and the each dataset's path
    """
    files = {}
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
    :return: glove csv
    """
    glove_file_path = join(os.getcwd(), "resources", properties["embeddings_file"])
    return pd.read_csv(glove_file_path, delimiter=" ", header=None)


def check_file_exists(directory, filename):
    """
    Checks if the the path of a file exists in order not to create it again.

    :param directory:
    :param filename: filenames
    :return: boolean true or false
    """
    path = join(os.getcwd(), directory, filename)
    return True if exists(path) else False


def write_to_pickle(obj, directory, filename):
    """
    Writes the user's vector after creating it to the output folder.

    :param obj: user's vector
    :param directory: output folder
    :param filename: the name of the folder where this vector is saved
    :return: the user ratings list
    """
    path = join(os.getcwd(), directory, filename)
    pickle.dump(obj, path)


def load_from_pickle(directory, file):
    """
    Loads the user's vector from the output folder by using the path.

    :param directory: output folder
    :param file: the ratings list from the output folder having this name
    :return: the vector of a user from the user ratings list
    """
    path = join(os.getcwd(), directory, file)
    return pickle.load(path)
