import utils
from preprocessing import data_preprocessing as dp
from models import classifiers


def main():
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
        for model in properties["models"]:
            if model != "kmeans":
                classifier, matrices = classifiers.run_cross_validation(model, properties, input_train, ratings_train,
                                                                        folds)
                # TODO run test

        print("Done!")

if __name__ == '__main__':
    main()
