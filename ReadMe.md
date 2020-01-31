# Filmoid

## Project configuration

The first set of options are used so as to setup the necessary folders which are the Datasets folder which 
contains the csv files with the datasets to be used, the resources folder which contains the word embeddings 
file and the output directory.
 
The word embeddings and dataset information is also used in the preprocessing step. In the output directory, the input
vectors are stored, therefore if the files are found they are loaded and the preprocessing step is skipped.

Finally, there are properties regarding the models configuration (e.g. number of neighbors in KNN etc).

Below, some information on the properties is presented:

* os: define your operating system. The possible values that are currently handled by the software are linux
and windows
* setup_folders: define whether you want to download the input data (glove file and datasets). If the files
are already downloaded inside the Datasets and resources directories, set this parameter as False.
* output_folder: the name of the output folder where the saved models will be stored as well as the results
* datasets_folder: the name of the datasets directory
* resources_folder: the name of the directory where the word embeddings file will be stored
* embeddings_file: the name of the word embeddings file to be used(e.g. "glove.6B.50d.txt")
* embeddings\_zip\_file: the zip file name to extract when downloaded (e.g. "glove.6B.zip")
* embeddings\_file\_url: the URL from where to download the word embeddings file (e.g. "http://nlp.stanford.edu/data/glove.6B.zip")
* dataset: the dataset to be used in the experiments. Currently the possible values are small and latest
* filenames: the csv names which you should provide as follows: ["links", "movies", "ratings", "tags"]
* dataset-file-extention: the file extension of the dataset files. You should provide it: ".csv"
* methods: the recommendation policies which are: ["collaborative", "content-based"]
* models:
    * collaborative: classifier to be used. ["k-means"]
    * content-based: the classifiers to be used. Choose some or all of the following: ["knn", "rf", "dnn"]
* aggregation: word embeddings aggregation strategies. Possible strategies: avg, max
* classification: binary or multi-class classification. Possible values: binary, multi
* cross-validation: the number of the folds for the k-fold cross-validation
* knn: configuration parameters for KNN model. The only configuration parameter so far is the neighbors paramter
* rf: configuration parameters for Random Forest model. You can configure the number of estimators and the max depth
* dnn: Deep Neural Network configuration:
    * hidden_layers: list of tuples which should provide the number of hidden units, the activation function and the Dropout value
    * sgd: stochastic gradient descent. Configuration parameters are: lr (learning rate), decay, momentum and nesterov
    * loss: loss function of the network. Two possible values: categorical_crossentropy or binary_crossentropy
    * metrics: network validation metrics
    * epochs: configure the number of epochs
    * batch_size: configuration parameter of the batch size

## Project execution

The main.py is used to execute all the workflow. As already mentioned, the first step is to provide a configuration
file. You may use the example_properties.yaml or you can create a new file in the same directory with the name
properties.yaml. Remember to include all the aforementioned parameters and change their values accordingly.

If you select to setup the directories, the program will execute one of the two scripts (setup.sh or setup.bat) based
on the operating system. The setup step will download the requested glove file and store it in the resources folder
as well as will download the necessary datasets from movielens ([F. Maxwell Harper and Joseph A. Konstan. 2015. 
The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems](https://doi.org/10.1145/2827872))

If the input vectors are not stored in pickle files, the program will generate the input data by executing the
preprocessing step. The step is different for the collaborative and content-based methods. For the first method, the
generated vectors have the same size as the number of existing movies in the dataset and for each instance they contain
the existing ratings of a user (if a rating does not exist the cell remains empty). The input vector for the content-based
approach is concatenated user and movie information, where each instance is associate with a line in the ratings.csv 
dataset. The respective labels are the ratings, which are also preprocessed by taking into account the classification
kind (binary or multi-class). 

Finally, the configuration file must include both the recommendation system methods (collaborative and/or content-based)
that the user wishes to execute as well as the respective models for each aforementioned method. The implemented models
for each method are the ones listed below:

1. Collaborative method:
    a. kmeans
2. Content-Based method:
    a. K-Nearest Neighbors
    b. Random Forest
    c. Deep Neural Networks