#!/usr/bin/env bash

# go to parent directory
cd ..

# create missing directories
mkdir -p "${OUTPUT_FOLDER}"
mkdir -p "${RESOURCES_FOLDER}"
mkdir -p "${DATASETS_FOLDER}"

# download glove file
cd "${RESOURCES_FOLDER}"
wget "${EMBEDDINGS_FILE_URL}"
unzip "${EMBEDDINGS_ZIP_FILE}"
rm "${EMBEDDINGS_ZIP_FILE}"
cd ..

# download datasets
cd "${DATASETS_FOLDER}"
wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip

unzip ml-latest-small.zip
unzip ml-latest.zip
rm ml-latest-small.zip ml-latest.zip
cd ..
