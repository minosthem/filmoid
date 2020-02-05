import zipfile
from os import getcwd, mkdir, chdir, remove
from os.path import join, exists

import wget
import yaml


def load_properties():
    print("Loading properties")
    properties_file = join(getcwd(), "properties", "properties.yaml")
    with open(properties_file, 'r') as f:
        return yaml.safe_load(f)


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


if __name__ == '__main__':
    properties = load_properties()
    create_missing_directories(properties["output_folder"], properties["resources_folder"],
                               properties["datasets_folder"])
    download_embeddings_file(properties["resources_folder"], properties["embeddings_file_url"])
    download_datasets(properties["datasets_folder"])
