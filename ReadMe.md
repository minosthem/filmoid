# Movie-Recommendation-System

## Properties explained

The first set of options are used so as to setup the necessary folders which are the Datasets folder which 
contains the csv files with the datasets to be used, the resources folder which contains the word embeddings 
file and the output directory. Below, some information on the properties is presented:

* os: define your operating system. The possible values that are currently handled by the software are linux
and windows
setup_folders: define whether you want to download the input data (glove file and datasets). If the files
are already downloaded inside the Datasets and resources directories, set this parameter as False.
output_folder: the name of the output folder where the saved models will be stored as well as the results
datasets_folder: the name of the datasets directory
resources_folder: the name of the directory where the word embeddings file will be stored
embeddings_file: the name of the word embeddings file to be used(e.g. "glove.6B.50d.txt")
embeddings_zip_file: the zip file name to extract when downloaded (e.g. "glove.6B.zip")
embeddings_file_url: the URL from where to download the word embeddings file (e.g. "http://nlp.stanford.edu/data/glove.6B.zip")
dataset: the dataset to be used in the experiments. Currently the possible values are small and latest
filenames: the csv names which you should provide as follows: ["links", "movies", "ratings", "tags"]
dataset-file-extention: the file extension of the dataset files. You should provide it: ".csv"
methods: the recommendation policies which are: ["collaborative", "content-based"]
models: the classifiers to be used. Choose some or all of the following: ["kmeans", "knn", "rf", "dnn"]
aggregation: word embeddings aggregation strategies. Possible strategies: avg, max
classification: binary or multi-class classification. Possible values: binary, multi