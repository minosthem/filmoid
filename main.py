import utils
from preprocessing import data_preprocessing as dp


def main():
    # load properties
    properties = utils.load_properties()
    if properties["setup_folders"]:
        utils.setup_folders(properties)
    # get dataset filenames to read
    file_names = utils.get_filenames(properties)
    # read datasets
    csvs = dp.read_csv(file_names)
    if "collaborative" in properties["methods"]:
        input_collaborative = dp.preprocessing_collaborative(properties, csvs)
    if "content-based" in properties["methods"]:
        input_content_based = dp.preprocessing_content_based(properties, csvs)


if __name__ == '__main__':
    main()
