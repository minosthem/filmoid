import time

import numpy as np

import utils
from preprocessing.collaborative_preprocessing import CollaborativePreprocessing
from preprocessing.content_based_preprocessing import ContentBasedPreprocessing
from preprocessing.data_preprocessing import DataPreprocessing


def run_collaborative(properties, csvs):
    """
    It processes the data to obtain the input vectors for the collaborative method and then uses the input data to
    create the model for the collaborative method.

    Args
        properties (dict): dictionary containing all the properties loaded from the yaml file - here the models
        parameter is used
        csvs (dict): the datasets loaded from the csv files
    """
    dp = CollaborativePreprocessing()
    print("Creating input vectors for collaborative method")
    dp.preprocess(properties=properties, datasets=csvs)
    input_data = dp.users_ratings
    for model in properties["models"]["collaborative"]:
        pass
        # TODO kmeans


def run_content_based(properties, csvs):
    """
    It processes the data to obtain the input vectors for the content-based methods and then uses them to create the
    models. It splits the data into train and test datasets, uses k-fold cross-validation and finally, run the models
    and write them into files for both the train and test results. In the end it calculates the average of the folds
    for the validation and test dataset.

    Args
        properties (dict): datasets, classification, models and output folder
        csvs (dict): the non-processed datasets
    """
    dp = ContentBasedPreprocessing()
    print("Creating input vectors for content-based method")
    dp.preprocess(properties, csvs)
    input_data = dp.input_data
    ratings = dp.ratings
    print("Split train and test datasets")
    input_train, input_test, ratings_train, ratings_test = dp.create_train_test_data(input_data, ratings)
    ratings_test = np.asarray(ratings_test)
    print("Get k-fold indices")
    folds = dp.create_cross_validation_data(input_train, properties)
    folds = list(folds)
    results_folder = "results_{}_{}".format(properties["dataset"], properties["classification"])
    classifiers = {}
    for model in properties["models"]["content-based"]:
        tic = time.time()
        classifier = utils.init_classifier(model)
        classifier.run_cross_validation(classifier, properties, input_train, ratings_train,
                                        folds)
        print("Time needed for classifier {} for train/test is {}".format(model, utils.elapsed_str(tic)))
        classifiers[model] = classifier
    print("Calculating average for macro/micro precision, recall and F-measure")
    for model in properties["models"]["content-based"]:
        classifier = classifiers[model]
        classifier.get_fold_avg_result(output_folder=properties["output_folder"], results_folder=results_folder)
        print("Best classifier with metric {} for model {}".format(properties["metric_best_model"], model))
        classifier.find_best_model(properties)
        true_labels, predicted_labels = classifier.test(input_test, ratings_test, kind="test")
        classifier.get_results(true_labels, predicted_labels, kind="test")
        classifier.write_test_results_to_file(properties["output_folder"], results_folder)
    # TODO visualize the results
    print("Done!")


def main():
    """
    Main function is the starting point of the program. Properties from the yaml are loaded in order to be used as
    configuration to the program. If the dataset and the embeddings files are not already downloaded, property
    setup_folders should be set to True. The scripts setup.sh or setup.bat are used (based on the operating system) in
    order to download the necessary files.

    The next step is to read the csv of the selected dataset and convert the data into input vectors.
    The recommendation is system is built using two methods: the collaborative and the content-based methods. For the
    collaborative method, the ratings of a user are used as input vector and they are fed in a kmeans model. On the
    other hand, for the content-based method, information about the user and the movie are used (associated with each
    live of the ratings.csv). The rating serves as label to each input vector. Based on the classification property,
    the labels are either binary (like, dislike) or 5 different classes (representing the possible ratings). The models
    used for the content-based method are: KNN, Random Forest and Deep Neural Networks.

    The dataset is split into training and testing datasets. The testing dataset is kept aside, while the training is
    split into folds for 10-fold cross-validation. Finally, the testing dataset is used as additional validation of the
    model. The confusion matrices of the folds as well as the one produced by the test dataset are used in order to
    calculate micro/macro precision, recall, F-measure. The visualization method can be used in order to produce plots
    of the micro/macro metrics.
    """
    # load properties
    print("Loading properties")
    properties = utils.load_properties()
    if properties["setup_folders"]:
        print("Set up folders is true. Glove vectors and datasets will be downloaded")
        utils.setup_folders(properties)
    # get dataset filenames to read
    print("Collect the dataset filenames")
    file_names = utils.get_filenames(properties)
    # read datasets
    print("Creating dataframes from the csvs in the selected dataset")
    dp = DataPreprocessing()
    dp.read_csv(file_names)
    csvs = dp.datasets
    if "collaborative" in properties["methods"]:
        run_collaborative(properties, csvs)
    if "content-based" in properties["methods"]:
        run_content_based(properties, csvs)


if __name__ == '__main__':
    main()
