#! /bin/bash

cd properties && cp example_properties.yaml properties.yaml && cd ..
# change the embeddings file in the properties.yaml if you wish to download a different file
# Recommendation: keep the same datasets that will be downloaded, otherwise the new dataset
# you wish to use should be in the same format (directory names, csv file names etc)
python env_prep.py
