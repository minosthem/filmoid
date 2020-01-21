#!/usr/bin/env bash

# create missing directories
mkdir -p "${OUTPUT_FOLDER}"
mkdir -p "${RESOURCES_FOLDER}"
mkdir -p "${DATASETS_FOLDER}"

# download glove file
wget "${EMBEDDINGS_FILE_URL}"
unzip "${EMBEDDINGS_FILE}" -d "${RESOURCES_FOLDER}"
rm "${EMBEDDINGS_FILE}"

wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip

unzip ml-latest-small.zip -d "${DATASETS_FOLDER}"
unzip ml-latest.zip -d "${DATASETS_FOLDER}"
rm ml-latest-small.zip ml-latest.zip
