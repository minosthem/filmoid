import os
from os.path import join, exists

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
    if prop["dataset"] == "latest":
        for file in prop["extrafiles"]:
            filename = file + prop["dataset-file-extention"]
            files[file] = join(dataset_path, filename)
    return files