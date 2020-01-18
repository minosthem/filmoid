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
    Load yaml file containing program's properties
    :return: the properties dictionary and the output folder path
    """
    file = properties_file if exists(properties_file) else example_properties_file
    with open(file, 'r') as f:
        properties = yaml.safe_load(f)
    return properties


def get_filenames(prop):
    files = {}
    base_path = join(os.getcwd(), datasets_folder)
    dataset_path = join(base_path, ml_latest_small_folder) if prop["dataset"] == "small" \
        else join(base_path, ml_latest)
    for file in prop["filenames"]:
        filename = file + prop["dataset-file-extention"]
        files[file] = join(dataset_path, filename)
    return files


def load_glove_file(properties):
    glove_file_path = join(os.getcwd(), "resources", properties["embeddings_file"])
    return pd.read_csv(glove_file_path, delimiter=" ", header=None)


def check_file_exists(directory, filename):
    path = join(os.getcwd(), directory, filename)
    return True if exists(path) else False


def write_to_pickle(object, directory, filename):
    path = join(os.getcwd(), directory, filename)
    pickle.dump(object, path)


def load_from_pickle(directory, file):
    path = join(os.getcwd(), directory, file)
    return pickle.load(path)
