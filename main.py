import utils
from models import classifiers, results
from preprocessing import data_preprocessing as dp


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
    csvs = dp.read_csv(file_names)
    if "collaborative" in properties["methods"]:
        print("Creating input vectors for collaborative method")
        input_data = dp.preprocessing_collaborative(properties, csvs)
        # TODO
    if "content-based" in properties["methods"]:
        print("Creating input vectors for content-based method")
        input_data, ratings = dp.preprocessing_content_based(properties, csvs)
        print("Split train and test datasets")
        input_train, input_test, ratings_train, ratings_test = dp.create_train_test_data(input_data, ratings)
        print("Get k-fold indices")
        folds = dp.create_cross_validation_data(input_train, properties)
        res = {}
        test_res = {}
        for model in properties["models"]:
            if model != "kmeans":
                classifier, matrices = classifiers.run_cross_validation(model, properties, input_train, ratings_train,
                                                                        folds)
                for i, matrix in enumerate(matrices):
                    res = results.write_results_to_file(properties, "fold_{}".format(i), model, matrix, res)
                conf_matrix = classifier.test(input_test, ratings_test)
                test_res = results.write_results_to_file(properties, "test_results", model, conf_matrix, test_res)
        # TODO visualize the results
        print("Done!")


if __name__ == '__main__':
    main()
